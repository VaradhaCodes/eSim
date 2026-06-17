"""Qt-free Icarus Verilog backend: compile + simulate.

Both the Verilog Simulator IDE (syntax check / simulate-and-wave) and the
d_cosim build shell out to ``iverilog``/``vvp`` in nearly the same way. This
module is the single, PyQt-free place that actually runs them, returning plain
result objects so the UI layer only deals with files, logging and widgets.

Keeping this synchronous and Qt-free is deliberate: callers that must stay
responsive run these functions on a worker thread (see hdl.jobs), and the pure
shape makes them unit-testable (integration tests skip when iverilog is absent).
"""
import os
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


@dataclass
class CompileResult:
    """Outcome of an ``iverilog`` invocation."""
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    out_path: Optional[str]          # compiled artifact, or None on failure
    cmd: List[str] = field(default_factory=list)
    written: List[str] = field(default_factory=list)   # source files written

    @property
    def output(self) -> str:
        """Combined stdout+stderr, for logging."""
        return (self.stdout or "") + (self.stderr or "")


@dataclass
class SimResult:
    """Outcome of a ``vvp`` run."""
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    vcd_path: Optional[str]          # produced VCD, or None if none written


def run_iverilog(
    iverilog_bin: str,
    src_paths: Sequence[str],
    out_path: str,
    *,
    cwd: Optional[str] = None,
    std: str = "-g2012",
    warnings: bool = False,
    extra_flags: Sequence[str] = (),
    timeout: Optional[float] = None,
) -> CompileResult:
    """Invoke iverilog on already-on-disk sources, producing ``out_path``.

    The lowest-level entry point, shared by :func:`compile_design` (IDE) and the
    d_cosim build. ``ok`` requires both a zero exit AND the artifact actually
    existing, so "compiler said 0 but wrote nothing" is reported as failure."""
    cmd = [iverilog_bin, std]
    if warnings:
        cmd.append("-Wall")
    cmd += list(extra_flags)
    cmd += ["-o", out_path] + list(src_paths)

    proc = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    ok = proc.returncode == 0 and os.path.isfile(out_path)
    return CompileResult(
        ok=ok, returncode=proc.returncode, stdout=proc.stdout,
        stderr=proc.stderr, out_path=out_path if ok else None,
        cmd=cmd, written=list(src_paths))


def compile_design(
    iverilog_bin: str,
    sources: Sequence[Tuple[str, str]],
    workdir: str,
    *,
    std: str = "-g2012",
    warnings: bool = False,
    extra_flags: Sequence[str] = (),
    out_name: str = "out.bin",
    timeout: Optional[float] = None,
) -> CompileResult:
    """Write ``sources`` into ``workdir`` and compile them with iverilog.

    ``sources`` is an ordered sequence of ``(filename, content)``; each is
    written under ``workdir`` with that exact name so iverilog's diagnostics
    reference the caller's filenames (the IDE relies on this to map an error
    back to the right editor tab). Returns a :class:`CompileResult`; never
    raises for an ordinary compile failure (only a timeout/OS error surfaces).
    """
    written: List[str] = []
    for fname, content in sources:
        path = os.path.join(workdir, fname)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        written.append(path)

    return run_iverilog(
        iverilog_bin, written, os.path.join(workdir, out_name),
        cwd=workdir, std=std, warnings=warnings, extra_flags=extra_flags,
        timeout=timeout)


def simulate(
    vvp_bin: str,
    out_path: str,
    workdir: str,
    *,
    env: Optional[dict] = None,
    vcd_name: str = "sim_out.vcd",
    timeout: Optional[float] = None,
) -> SimResult:
    """Run a compiled design under ``vvp`` in ``workdir``.

    Returns a :class:`SimResult`; ``vcd_path`` is set only when the run
    produced ``vcd_name`` (so the caller can tell "no $dumpfile" apart from a
    crash). ``env`` lets the caller fix up the loader path (e.g. prepend the
    vvp dir to PATH on Windows for its DLLs)."""
    proc = subprocess.run(
        [vvp_bin, out_path], cwd=workdir, env=env,
        capture_output=True, text=True, timeout=timeout)
    vcd = os.path.join(workdir, vcd_name)
    return SimResult(
        ok=proc.returncode == 0, returncode=proc.returncode,
        stdout=proc.stdout, stderr=proc.stderr,
        vcd_path=vcd if os.path.isfile(vcd) else None)


def vvp_env(vvp_bin: str, base_env: Optional[dict] = None,
            libdir: Optional[str] = None) -> dict:
    """Build an environment for running vvp so its shared library (libvvp)
    resolves regardless of how Icarus was installed.

    Prepends ``libdir`` (the iverilog ``lib/`` holding libvvp) to the OS dynamic
    loader path -- PATH on Windows, LD_LIBRARY_PATH elsewhere -- which is what a
    prefix build (e.g. ~/iverilog) needs but is not on the default loader path.
    On Windows the directory holding vvp itself is also added for its MinGW
    DLLs. Pass ``libdir`` from CosimConfig.iverilog_libdir()."""
    env = dict(base_env if base_env is not None else os.environ)
    loader = 'PATH' if os.name == 'nt' else 'LD_LIBRARY_PATH'
    extra = []
    if libdir and os.path.isdir(libdir):
        extra.append(libdir)
    if os.name == 'nt' and vvp_bin:
        extra.append(os.path.dirname(os.path.abspath(vvp_bin)))
    if extra:
        existing = env.get(loader, "")
        env[loader] = os.pathsep.join(extra + ([existing] if existing else []))
    return env
