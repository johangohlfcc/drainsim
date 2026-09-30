"""Off-screen 3D rendering of the world-frame VTK series (needs ``pip install vtk``
and ffmpeg). Produces an MP4 or GIF in which the object moves through a
fixed, semi-transparent bath.

    from drainsim.worldviz import export_world
    from drainsim.render3d import render_animation
    export_world(sim, hist, "out/vtk", prefix="box")
    render_animation("out/vtk", "box", "out/box.mp4")
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import numpy as np


def read_pvd(path):
    root = ET.parse(path).getroot()
    base = os.path.dirname(path)
    return [(float(d.get("timestep")), os.path.join(base, d.get("file")))
            for d in root.iter("DataSet")]


def _actor(vtk, color, opacity, reader=None, polydata=None, specular=0.2):
    m = vtk.vtkPolyDataMapper()
    if reader is not None:
        m.SetInputConnection(reader.GetOutputPort())
    else:
        m.SetInputData(polydata)
    m.ScalarVisibilityOff()
    a = vtk.vtkActor()
    a.SetMapper(m)
    p = a.GetProperty()
    p.SetColor(*color)
    p.SetOpacity(opacity)
    p.SetSpecular(specular)
    p.SetSpecularPower(20)
    return a


def render_animation(outdir, prefix, path, fps=15, size=(1024, 720),
                     azimuth=-55.0, elevation=22.0, zoom=1.25,
                     object_opacity=0.35, show_air=True, keep_frames=False,
                     show_film=True, film_max_um=60.0, show_bath=True,
                     fit="all", focus=None, frames=None):
    """Render ``<prefix>_*.pvd`` written by ``worldviz.export_world``.

    ``fit="object"`` frames the camera on the object only (not the bath);
    ``focus=(lo, hi)`` frames a world box instead. ``frames`` selects frame
    indices; if ``path`` ends in ``.png`` the (last selected) frame is saved
    as a still image.

    If a film series exists (run with ``film=True``) and ``show_film``, the
    object is drawn as its film carrier coloured by film thickness
    (0..``film_max_um`` micrometres) instead of plain grey.
    """
    import vtk

    obj = read_pvd(os.path.join(outdir, f"{prefix}_object.pvd"))
    liq = read_pvd(os.path.join(outdir, f"{prefix}_liquid.pvd"))
    air_f = os.path.join(outdir, f"{prefix}_air.pvd")
    air = read_pvd(air_f) if (show_air and os.path.exists(air_f)) else None

    ren = vtk.vtkRenderer()
    ren.SetBackground(1, 1, 1)
    ren.SetBackground2(0.86, 0.89, 0.93)
    ren.GradientBackgroundOn()
    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1)
    win.SetSize(*size)
    win.SetAlphaBitPlanes(1)
    win.SetMultiSamples(0)
    win.AddRenderer(ren)
    ren.SetUseDepthPeeling(1)
    ren.SetMaximumNumberOfPeels(8)

    rb = vtk.vtkXMLPolyDataReader()
    rb.SetFileName(os.path.join(outdir, f"{prefix}_bath.vtp"))
    rb.Update()
    if show_bath:
        ren.AddActor(_actor(vtk, (0.55, 0.74, 0.93), 0.22, reader=rb, specular=0.0))
    readers = {}
    film_f = os.path.join(outdir, f"{prefix}_film.pvd")
    film = read_pvd(film_f) if (show_film and os.path.exists(film_f)) else None
    for name, series, color, op in (("object", None if film else obj,
                                     (0.72, 0.73, 0.76), object_opacity),
                                    ("liquid", liq, (0.10, 0.33, 0.78), 1.0),
                                    ("air", air, (1.0, 0.60, 0.12), 1.0)):
        if series is None:
            continue
        r = vtk.vtkXMLPolyDataReader()
        r.SetFileName(series[0][1])
        readers[name] = (r, series)
        ren.AddActor(_actor(vtk, color, op, reader=r))
    if film:
        import matplotlib
        cm = matplotlib.colormaps["plasma"]
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        for q in range(256):
            rgba = cm(q / 255.0)
            lut.SetTableValue(q, rgba[0], rgba[1], rgba[2], 1.0)
        lut.SetBelowRangeColor(0.72, 0.73, 0.76, 1.0)
        lut.UseBelowRangeColorOn()
        lut.SetTableRange(0.05, film_max_um)
        lut.Build()
        rf = vtk.vtkXMLPolyDataReader()
        rf.SetFileName(film[0][1])
        readers["film"] = (rf, film)
        mf = vtk.vtkPolyDataMapper()
        mf.SetInputConnection(rf.GetOutputPort())
        mf.SetScalarModeToUseCellFieldData()
        mf.SelectColorArray("film_um")
        mf.SetLookupTable(lut)
        mf.SetScalarRange(0.05, film_max_um)
        mf.UseLookupTableScalarRangeOn()
        mf.ScalarVisibilityOn()
        af = vtk.vtkActor()
        af.SetMapper(mf)
        af.GetProperty().SetOpacity(max(object_opacity, 0.55))
        ren.AddActor(af)
        bar = vtk.vtkScalarBarActor()
        bar.SetLookupTable(lut)
        bar.SetTitle("film [um]")
        bar.SetNumberOfLabels(4)
        bar.SetLabelFormat("%.0f")
        bar.SetOrientationToHorizontal()
        bar.SetPosition(0.62, 0.03)
        bar.SetWidth(0.33)
        bar.SetHeight(0.09)
        for tp in (bar.GetTitleTextProperty(), bar.GetLabelTextProperty()):
            tp.SetColor(0.1, 0.1, 0.1)
            tp.SetFontSize(14)
            tp.ItalicOff()
            tp.ShadowOff()
        ren.AddViewProp(bar)

    # camera: fit the bath box plus the swept object
    bounds = np.array(rb.GetOutput().GetBounds()).reshape(3, 2)
    if fit == "object":
        bounds = np.array([[np.inf, -np.inf]] * 3)
    r0 = vtk.vtkXMLPolyDataReader()
    for _, fn in obj[:: max(1, len(obj) // 20)]:
        r0.SetFileName(fn)
        r0.Update()
        b = np.array(r0.GetOutput().GetBounds()).reshape(3, 2)
        if np.all(np.isfinite(b)) and b[0, 0] <= b[0, 1]:
            bounds[:, 0] = np.minimum(bounds[:, 0], b[:, 0])
            bounds[:, 1] = np.maximum(bounds[:, 1], b[:, 1])
    if focus is not None:
        bounds = np.array(focus, float).T.copy()
    ren.ResetCamera(*bounds.ravel())
    cam = ren.GetActiveCamera()
    cam.SetViewUp(0, 0, 1)
    c = bounds.mean(1)
    cam.SetFocalPoint(*c)
    cam.SetPosition(c[0], c[1] - 1.0, c[2])
    cam.Azimuth(azimuth)
    cam.Elevation(elevation)
    ren.ResetCamera(*bounds.ravel())
    cam.Zoom(zoom)

    txt = vtk.vtkTextActor()
    txt.GetTextProperty().SetFontSize(20)
    txt.GetTextProperty().SetColor(0.1, 0.1, 0.1)
    txt.SetPosition(15, 15)
    ren.AddViewProp(txt)

    tmp = tempfile.mkdtemp(prefix="drainsim_frames_")
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(win)
    png = vtk.vtkPNGWriter()
    png.SetInputConnection(w2i.GetOutputPort())
    sel = range(len(obj)) if frames is None else list(frames)
    for i in sel:
        t = obj[i][0]
        for name, (r, series) in readers.items():
            r.SetFileName(series[i][1])
            r.Update()
        txt.SetInput(f"t = {t:6.2f} s")
        win.Render()
        w2i.Modified()
        png.SetFileName(os.path.join(tmp, f"f_{i:05d}.png"))
        png.Write()

    if path.endswith(".png"):
        shutil.copy(os.path.join(tmp, f"f_{sel[-1]:05d}.png"), path)
        shutil.rmtree(tmp, ignore_errors=True)
        return path
    if frames is not None:                       # renumber for ffmpeg
        for k, i in enumerate(sel):
            os.rename(os.path.join(tmp, f"f_{i:05d}.png"),
                      os.path.join(tmp, f"g_{k:05d}.png"))
        pattern = os.path.join(tmp, "g_%05d.png")
    else:
        pattern = os.path.join(tmp, "f_%05d.png")
    if path.endswith(".gif"):
        pal = os.path.join(tmp, "palette.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                        "-i", pattern, "-vf", "palettegen", pal], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                        "-i", pattern, "-i", pal, "-lavfi", "paletteuse", path],
                       check=True)
    else:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                        "-i", pattern, "-pix_fmt", "yuv420p", "-vf",
                        "pad=ceil(iw/2)*2:ceil(ih/2)*2", path], check=True)
    if keep_frames:
        return path, tmp
    shutil.rmtree(tmp, ignore_errors=True)
    return path
