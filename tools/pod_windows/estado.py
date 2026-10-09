"""Estado de la PC prestada: trabajos lanzados (vivos o no, RAM de todo su árbol de procesos), RAM libre del dueño
y fin de cada log. `--detener NOMBRE` termina el trabajo entero (el python.exe del venv es un lanzador que arranca el
intérprete real como hijo: hay que terminar el árbol).

  C:\\hxpod\\venv\\Scripts\\python.exe C:\\hxpod\\estado.py [--detener NOMBRE] [--lineas N]
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from pathlib import Path

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
RUNS = Path(r"C:\hxpod\runs")
STILL_ACTIVE = 259
TH32CS_SNAPPROCESS = 0x2
k32.OpenProcess.restype = wintypes.HANDLE
k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
        (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                       "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                                       "PagefileUsage", "PeakPagefileUsage")]


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_wchar * 260)]


def children_map():
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    out = {}
    e = PROCESSENTRY32W()
    e.dwSize = ctypes.sizeof(e)
    ok = k32.Process32FirstW(snap, ctypes.byref(e))
    while ok:
        out.setdefault(e.th32ParentProcessID, []).append(e.th32ProcessID)
        ok = k32.Process32NextW(snap, ctypes.byref(e))
    k32.CloseHandle(snap)
    return out


def tree(pid, kids):
    todo, seen = [pid], []
    while todo:
        p = todo.pop()
        if p not in seen:
            seen.append(p)
            todo += kids.get(p, [])
    return seen


def info(pid):
    """(vivo, working set, pico) de un proceso."""
    h = k32.OpenProcess(0x1000 | 0x0400 | 0x0010 | 0x0001, False, pid)
    if not h:
        return False, 0, 0
    code = wintypes.DWORD()
    alive = bool(k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == STILL_ACTIVE
    pmc = PROCESS_MEMORY_COUNTERS()
    pmc.cb = ctypes.sizeof(pmc)
    k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
    ws = pk = 0
    if alive and k32.K32GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
        ws, pk = pmc.WorkingSetSize, pmc.PeakWorkingSetSize
    k32.CloseHandle(h)
    return alive, ws, pk


def kill(pid):
    h = k32.OpenProcess(0x0001, False, pid)
    if h:
        k32.TerminateProcess(h, 1)
        k32.CloseHandle(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detener", default=None)
    ap.add_argument("--lineas", type=int, default=3)
    a = ap.parse_args()
    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(m)
    k32.GlobalMemoryStatusEx(ctypes.byref(m))
    print(f"PC: RAM libre {m.ullAvailPhys / 2**30:.1f} de {m.ullTotalPhys / 2**30:.1f} GB (uso {m.dwMemoryLoad}%)")
    kids = children_map()
    for pidf in sorted(RUNS.glob("*.pid")):
        pid = int(pidf.read_text())
        procs = tree(pid, kids)
        stats = [info(p) for p in procs]
        alive = any(s[0] for s in stats)
        ws = sum(s[1] for s in stats) / 2 ** 30
        pk = sum(s[2] for s in stats) / 2 ** 30
        extra = f", RAM {ws:.2f} GB (pico {pk:.2f}), {len(procs)} procesos" if alive else ""
        if alive and a.detener == pidf.stem:
            for p in reversed(procs):
                kill(p)
            alive = False
            extra += " -> DETENIDO"
        print(f"{pidf.stem}: pid {pid} {'corriendo' if alive else 'terminado'}{extra}")
        log = RUNS / f"{pidf.stem}.log"
        if log.exists():
            for line in (log.read_text(encoding="utf-8", errors="replace").splitlines()[-a.lineas:]
                         if a.lineas > 0 else []):
                print("   ", line[:300])


if __name__ == "__main__":
    main()
