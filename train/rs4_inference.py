"""Grouped deterministic inference with device-resident RS4 player memory.

Only observations and decisions cross PCIe on every decision. Recurrent state
stays on the policy's device, and sequence-boundary snapshots can remain there
for PPO. Sampling and actual executed actions are deliberately outside graphs.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import warnings

import torch

from .cuda_decisions import CudaDecisionGraph


class _MultiInputGraph:
    """Static-input graph for FF or (obs, memory, previous action, reset) step."""

    def __init__(self, device, mode):
        self.device, self.mode = device, mode
        self.graph = self.inputs = self.outputs = self.stream = None
        self.disabled_reason = None
        self.captures = self.replays = 0
        self.capture_seconds = 0.0

    @torch.no_grad()
    def run(self, forward, arguments, *, capacity=None, previous_sentinel=None):
        if self.disabled_reason is not None:
            return forward(*arguments)
        rows = len(arguments[0])
        capacity = rows if capacity is None else capacity
        if capacity < rows:
            raise ValueError("CUDA graph capacity cannot discard real rows")
        if self.graph is None:
            started = time.perf_counter()
            try:
                self.inputs = tuple(argument.new_empty((capacity, *argument.shape[1:])) for argument in arguments)
                self._copy_inputs(arguments, rows, previous_sentinel)
                current = torch.cuda.current_stream(self.device)
                self.stream = torch.cuda.Stream(device=self.device)
                self.stream.wait_stream(current)
                with torch.cuda.stream(self.stream):
                    for _ in range(3):
                        forward(*self.inputs)
                current.wait_stream(self.stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=self.stream):
                    outputs = forward(*self.inputs)
                current.wait_stream(self.stream)
                self.graph, self.outputs = graph, outputs
                self.captures += 1
            except RuntimeError as error:
                if self.mode != "auto" or not CudaDecisionGraph._unsupported(error):
                    raise
                self.disabled_reason = str(error)
                self.graph = self.inputs = self.outputs = self.stream = None
                warnings.warn(f"RS4 recurrent CUDA graph unavailable; eager inference: {error}", RuntimeWarning)
                return forward(*arguments)
            finally:
                self.capture_seconds += time.perf_counter() - started
        self._copy_inputs(arguments, rows, previous_sentinel)
        self.graph.replay()
        self.replays += 1
        # Padding never enters action sampling, trajectory masks or PPO counts.
        return tuple(output[:rows] for output in self.outputs)

    def _copy_inputs(self, arguments, rows, previous_sentinel):
        for index, (destination, source) in enumerate(zip(self.inputs, arguments)):
            if (destination.shape[1:] != source.shape[1:] or destination.dtype != source.dtype
                    or destination.device != source.device or len(source) != rows):
                raise ValueError("CUDA graph input layout changed without invalidation")
            destination[:rows].copy_(source)
            # A previously larger selected group may have left real data in
            # these slots. Restore neutral pad inputs, including memory reset.
            pad = destination[rows:]
            if index == 2 and previous_sentinel is not None:
                pad.fill_(previous_sentinel)
            elif index == 3:
                pad.fill_(True)
            else:
                pad.zero_()


@dataclass
class _Controller:
    model: torch.nn.Module
    total_rows: int
    memory: torch.Tensor | None
    previous_action: torch.Tensor | None
    episode_start: torch.Tensor | None


class RoutedPolicyInference:
    """Memory is independent for each controller and global player row.

    ``obs`` contains *selected* rows in the same order as ``indices``; indices
    address the controller's persistent global table. Reusing a row for a new
    player/controller requires ``reset_rows``. A controller must not be removed
    while a live match still uses it. The trainer owns those match boundaries.
    """

    def __init__(self, device, mode="auto", max_policies=4):
        if mode not in ("legacy", "eager", "auto", "graph"):
            raise ValueError(f"Unknown inference backend {mode}")
        if max_policies < 1 or max_policies > 4:
            raise ValueError("RS4 v3 permits one to four neural policies per rollout")
        self.device, self.mode, self.max_policies = torch.device(device), mode, max_policies
        self.controllers, self.graphs = {}, {}
        self.generation = 0
        self.graph_input_rows = self.graph_padding_rows = 0

    def register(self, key, model, total_rows):
        if total_rows < 1:
            raise ValueError("Controller row capacity must be positive")
        if key in self.controllers:
            controller = self.controllers[key]
            if controller.model is not model or controller.total_rows != total_rows:
                raise ValueError("Controller identity/capacity changed; unregister it at a match boundary")
            return
        if len(self.controllers) >= self.max_policies:
            raise ValueError("More than four active neural controllers would exceed the RS4 v3 contract")
        if next(model.parameters()).device != self.device:
            raise ValueError("All grouped policies must already reside on the inference device")
        recurrent = bool(getattr(model, "is_recurrent", False))
        memory = model.initial_state(total_rows) if recurrent else None
        previous = torch.full((total_rows,), model.n_actions, dtype=torch.long, device=self.device) if recurrent else None
        start = torch.ones(total_rows, dtype=torch.bool, device=self.device) if recurrent else None
        self.controllers[key] = _Controller(model, int(total_rows), memory, previous, start)

    def unregister(self, key):
        self.controllers.pop(key, None)
        self.graphs = {cache_key: graph for cache_key, graph in self.graphs.items() if cache_key[0] != key}

    def invalidate(self):
        """Call after PPO/normalizer changes; stale graphs are never replayed."""
        self.graphs.clear()
        self.generation += 1
        self.graph_input_rows = self.graph_padding_rows = 0

    def _indices(self, controller, indices):
        if indices is None:
            return torch.arange(controller.total_rows, device=self.device)
        result = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        if result.ndim != 1:
            raise ValueError("Player indices must be one-dimensional")
        # Range/uniqueness checking on CUDA would force a per-decision sync.
        # CPU checks are useful for tests; trainer routes guarantee this on CUDA.
        if self.device.type == "cpu" and result.numel():
            if result.min() < 0 or result.max() >= controller.total_rows or len(result.unique()) != len(result):
                raise ValueError("Player indices must be unique and within the controller table")
        return result

    def get_state(self, key, indices=None, *, clone=False):
        controller = self.controllers[key]
        if controller.memory is None:
            return None, None, None
        if indices is None:
            tensors = (controller.memory, controller.previous_action, controller.episode_start)
        else:
            selected = self._indices(controller, indices)
            tensors = tuple(tensor.index_select(0, selected) for tensor in
                            (controller.memory, controller.previous_action, controller.episode_start))
        return tuple(tensor.clone() for tensor in tensors) if clone else tensors

    @torch.no_grad()
    def infer(self, key, obs, indices=None, *, commit=True):
        controller = self.controllers[key]
        selected = self._indices(controller, indices)
        if obs.device != self.device or obs.ndim != 2 or len(obs) != len(selected):
            raise ValueError("obs must be selected rows on the controller device, matching indices")
        if not len(obs):
            return (obs.new_empty((0, controller.model.n_actions)), obs.new_empty(0),
                    obs.new_empty((0, controller.model.memory_size)) if controller.memory is not None else None)
        if controller.memory is None:
            arguments, forward = (obs,), controller.model
        else:
            memory, previous, start = self.get_state(key, selected)
            arguments, forward = (obs, memory, previous, start), controller.model.step
        if self.device.type == "cuda" and self.mode in ("auto", "graph"):
            # Matches finish asynchronously and change routed batch sizes.
            # Exact-shape graph caches would recapture dozens of shapes and
            # retain their pools in a single rollout. Power-of-two buckets
            # bound captures/memory without changing any useful PPO sample.
            capacity = 1 << (len(obs) - 1).bit_length()
            cache_key = (key, capacity, obs.shape[1], obs.dtype)
            graph = self.graphs.setdefault(cache_key, _MultiInputGraph(self.device, self.mode))
            result = graph.run(forward, arguments, capacity=capacity, previous_sentinel=controller.model.n_actions)
            self.graph_input_rows += len(obs)
            self.graph_padding_rows += capacity - len(obs)
        else:
            result = forward(*arguments)
        if controller.memory is None:
            logits, values = result
            return logits, values, None
        logits, values, memory = result
        if commit:
            controller.memory.index_copy_(0, selected, memory)
            controller.episode_start[selected] = False
        return logits, values, memory

    @torch.no_grad()
    def record_actions(self, key, indices, actions):
        """Must receive actions actually executed, including any external override."""
        controller = self.controllers[key]
        if controller.memory is None:
            return
        selected = self._indices(controller, indices)
        actual = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        if actual.shape != selected.shape:
            raise ValueError("Actual actions must match selected player rows")
        if self.device.type == "cpu" and actual.numel() and (actual.min() < 0 or actual.max() >= controller.model.n_actions):
            raise ValueError("Executed action is outside the policy action space")
        controller.previous_action.index_copy_(0, selected, actual)

    @torch.no_grad()
    def reset_rows(self, indices, key=None):
        for controller_key in (list(self.controllers) if key is None else [key]):
            controller = self.controllers[controller_key]
            if controller.memory is None:
                continue
            selected = self._indices(controller, indices)
            controller.memory[selected] = 0
            controller.previous_action[selected] = controller.model.n_actions
            controller.episode_start[selected] = True

    @property
    def capture_seconds(self):
        return sum(graph.capture_seconds for graph in self.graphs.values())

    @property
    def captures(self):
        return sum(graph.captures for graph in self.graphs.values())

    @property
    def replays(self):
        return sum(graph.replays for graph in self.graphs.values())

    @property
    def disabled_reason(self):
        return next((graph.disabled_reason for graph in self.graphs.values() if graph.disabled_reason), None)
