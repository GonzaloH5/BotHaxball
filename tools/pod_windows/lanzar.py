r"""Lanza un trabajo en la PC prestada (cuenta hxpod) con límites duros de Windows, sin ventanas y sin depender de SSH.

Límites (job object de Windows; no hace falta ser administrador y valen aunque el proceso se desboque):
* afinidad: solo los hilos lógicos indicados (por defecto 8–11 = los dos últimos núcleos físicos del i5-11400F);
* tope duro de CPU: `--cpu-pct` del total de la máquina (por defecto 33%);
* prioridad Idle: cualquier programa del dueño le gana siempre la CPU;
* memoria: tope duro de `--ram-gb` para todo el trabajo (si se pasa, falla el trabajo, no la PC) y prioridad de
  memoria baja (Windows le saca páginas a este proceso antes que a los del dueño);
* disco: no lanza si C:\hxpod ya ocupa más de 6 GB o si al disco le quedan menos de 8 GB libres. Cada entrenamiento
  guarda solo dos checkpoints que se pisan (`model.pt` y `ultimo.pt`, ~1,6 MB cada uno).

El proceso se separa del job de la sesión SSH (sigue corriendo al cerrarla), sin consola ni ventana. Salida en
`--log`; el pid queda en `C:\\hxpod\\runs\\<nombre>.pid`.

  call C:\\hxpod\\entorno.bat && C:\\hxpod\\venv\\Scripts\\python.exe C:\\hxpod\\lanzar.py --nombre l0 -- <comando...>
"""
from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import subprocess
import sys
from ctypes import wintypes
from pathlib import Path

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
RUNS = Path(r"C:\hxpod\runs")
MAX_GB = 6.0            # lo que puede ocupar C:\hxpod en total
MIN_FREE_GB = 8.0       # lo que siempre le queda libre al dueño en C:

CREATE_BREAKAWAY_FROM_JOB = 0x01000000
CREATE_NO_WINDOW = 0x08000000
IDLE_PRIORITY_CLASS = 0x00000040
JOB_OBJECT_LIMIT_AFFINITY = 0x00000010
JOB_OBJECT_LIMIT_PRIORITY_CLASS = 0x00000020
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
JobObjectExtendedLimitInformation = 9
JobObjectCpuRateControlInformation = 15
JOB_OBJECT_CPU_RATE_CONTROL_ENABLE = 0x1
JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP = 0x4
ProcessMemoryPriority = 0
MEMORY_PRIORITY_LOW = 2


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                                  "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD)]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION), ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


class JOBOBJECT_CPU_RATE_CONTROL_INFORMATION(ctypes.Structure):
    _fields_ = [("ControlFlags", wintypes.DWORD), ("CpuRate", wintypes.DWORD)]


class MEMORY_PRIORITY_INFORMATION(ctypes.Structure):
    _fields_ = [("MemoryPriority", wintypes.ULONG)]


def check(ok, what):
    if not ok:
        raise OSError(ctypes.get_last_error(), f"{what} falló")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nombre", required=True)
    ap.add_argument("--hilos", default="8,9,10,11", help="hilos lógicos permitidos")
    ap.add_argument("--cpu-pct", type=float, default=33.0, help="tope duro de CPU, % del total de la máquina")
    ap.add_argument("--ram-gb", type=float, default=8.0)
    ap.add_argument("--cwd", default=r"C:\hxpod\repo")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        sys.exit("falta el comando")
    # disco del dueño: C:\hxpod no pasa de MAX_GB y siempre le quedan al menos MIN_FREE_GB libres
    used = sum(f.stat().st_size for f in Path(r"C:\hxpod").rglob("*") if f.is_file()) / 2 ** 30
    free = shutil.disk_usage("C:\\").free / 2 ** 30
    if used > MAX_GB or free < MIN_FREE_GB:
        sys.exit(f"no se lanza: C:\\hxpod usa {used:.1f} GB (tope {MAX_GB}) y el disco tiene {free:.1f} GB libres "
                 f"(mínimo {MIN_FREE_GB})")
    RUNS.mkdir(parents=True, exist_ok=True)
    log = open(RUNS / f"{a.nombre}.log", "ab")
    env = dict(os.environ, OMP_NUM_THREADS=str(len(a.hilos.split(","))),
               MKL_NUM_THREADS=str(len(a.hilos.split(","))), NUMBA_NUM_THREADS=str(len(a.hilos.split(","))))

    job = k32.CreateJobObjectW(None, None)
    check(job, "CreateJobObject")
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = (JOB_OBJECT_LIMIT_AFFINITY | JOB_OBJECT_LIMIT_PRIORITY_CLASS
                                             | JOB_OBJECT_LIMIT_JOB_MEMORY)
    info.BasicLimitInformation.Affinity = sum(1 << int(h) for h in a.hilos.split(","))
    info.BasicLimitInformation.PriorityClass = IDLE_PRIORITY_CLASS
    info.JobMemoryLimit = int(a.ram_gb * 2 ** 30)
    check(k32.SetInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(info),
                                      ctypes.sizeof(info)), "SetInformationJobObject (límites)")
    rate = JOBOBJECT_CPU_RATE_CONTROL_INFORMATION(JOB_OBJECT_CPU_RATE_CONTROL_ENABLE | JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP,
                                                  int(a.cpu_pct * 100))
    check(k32.SetInformationJobObject(job, JobObjectCpuRateControlInformation, ctypes.byref(rate),
                                      ctypes.sizeof(rate)), "SetInformationJobObject (tope de CPU)")

    p = subprocess.Popen(cmd, cwd=a.cwd, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                         creationflags=CREATE_BREAKAWAY_FROM_JOB | CREATE_NO_WINDOW | IDLE_PRIORITY_CLASS)
    h = wintypes.HANDLE(int(p._handle))
    check(k32.AssignProcessToJobObject(job, h), "AssignProcessToJobObject")
    mp = MEMORY_PRIORITY_INFORMATION(MEMORY_PRIORITY_LOW)
    k32.SetProcessInformation(h, ProcessMemoryPriority, ctypes.byref(mp), ctypes.sizeof(mp))
    (RUNS / f"{a.nombre}.pid").write_text(str(p.pid))
    print(f"lanzado {a.nombre}: pid {p.pid}; hilos {a.hilos}, CPU <= {a.cpu_pct}%, RAM <= {a.ram_gb} GB, prioridad Idle")
    # el job sigue vivo mientras tenga procesos (no se usa KILL_ON_JOB_CLOSE)


if __name__ == "__main__":
    main()
