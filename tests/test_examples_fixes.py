"""Tests for the fixes in the example scripts from the code review of
drainsim 6.3. Run: python -m pytest -q tests/test_examples_fixes.py"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples")
if EXAMPLES not in sys.path:
    sys.path.insert(0, EXAMPLES)


def _movie_args(rec, out, **kw):
    """The options car_movie.render() reads, with the command-line defaults."""
    a = dict(render=str(rec), preview=None, workers=3, fps=30, speedup=2.0, speedup_late=10.0,
             slow_hang=10.0, size3d=[320, 240], width2d=200, ssaa=1, crf=20, preset="fast",
             car_opacity=0.25, drop_scale=3.0, cam_dist=10.0, azimuth=-55.0, elevation=14.0,
             look="hd", orbit=0.0, sim_dt=0.1, dt_hang=0.5, out=str(out))
    a.update(kw)
    return SimpleNamespace(**a)


def test_render_workers_get_the_look_and_orbit_options(tmp_path, monkeypatch):
    """car_movie --render --workers N started its workers without --look,
    --orbit, --sim-dt and --dt-hang, so a parallel movie had no camera orbit
    (the stills, drawn in the parent, had)."""
    pytest.importorskip("vtk")
    import car_movie
    rec = tmp_path / "rec"
    rec.mkdir()
    (rec / "meta.json").write_text(json.dumps(dict(t_dipout=35.0, t_end=155.0)))
    cmds = []

    class FakeProc:
        returncode = 0

        def __init__(self, cmd, **kw):
            cmds.append(list(cmd))

        def wait(self):
            return 0
    monkeypatch.setattr(car_movie.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(car_movie.subprocess, "run", lambda *a, **k: None)
    a = _movie_args(rec, tmp_path / "car.mp4", look="plain", orbit=25.0, sim_dt=0.2,
                    dt_hang=1.0)
    car_movie.render(a)
    assert len(cmds) == 3
    for c in cmds:
        def val(flag):
            return c[c.index(flag) + 1]
        assert val("--look") == "plain" and float(val("--orbit")) == 25.0
        assert float(val("--sim-dt")) == 0.2 and float(val("--dt-hang")) == 1.0
