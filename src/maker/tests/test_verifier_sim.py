"""End-to-end test of the Verilog Simulator IDE's async simulate path.

Drives VerilogVerifier.simulate_and_wave on its default design+testbench under a
real Qt event loop and asserts the simulationSucceeded signal fires. Skipped
unless a working iverilog+vvp is present (so CI without Icarus stays green).
"""
import pytest

from maker import CosimConfig

_IV = CosimConfig.iverilog_binary()
_VVP = CosimConfig.vvp_binary()
needs_sim = pytest.mark.skipif(
    not (_IV and _VVP and CosimConfig.has_iverilog()),
    reason="iverilog+vvp (with libvvp) not installed")


@needs_sim
def test_verifier_emits_simulation_succeeded(qapp):
    from PyQt6.QtCore import QEventLoop, QTimer
    from maker.VerilogVerifier import VerilogVerifier

    w = VerilogVerifier()
    loop = QEventLoop()
    fired = {"ok": False}

    def on_ok():
        fired["ok"] = True
        loop.quit()

    w.simulationSucceeded.connect(on_ok)
    # Kick the (async) run once the loop is spinning; bail after 20s.
    QTimer.singleShot(0, w.simulate_and_wave)
    QTimer.singleShot(20000, loop.quit)
    loop.exec()

    assert fired["ok"], "default counter design should compile+simulate cleanly"
    w.deleteLater()
