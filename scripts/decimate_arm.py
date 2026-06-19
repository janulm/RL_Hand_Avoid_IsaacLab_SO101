"""Decimate the human-arm obstacle mesh to a low-poly version.

The original arm.usd is ~1M triangles (~76 MB) of pure geometry with no image
textures -- massively overkill for a 100x100 policy camera. This welds and
quadric-decimates the mesh to a target triangle count while keeping the original
OmniPBR material (so the skin color/lighting is unchanged), and writes a new USD.

Run inside the container (needs the Kit runtime for `pxr`):

    ./docker/run.sh bash -c "python -m pip install -q fast-simplification && \
        python scripts/decimate_arm.py --target_tris 8000"
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--target_tris", type=int, default=8000)
parser.add_argument(
    "--src",
    default="/workspace/RL_Hand_Avoid_IsaacLab_SO101/source/so_arm101_avoid/so_arm101_avoid/tasks/reach_avoid/assets/arm.usd",
)
parser.add_argument(
    "--dst",
    default="/workspace/RL_Hand_Avoid_IsaacLab_SO101/source/so_arm101_avoid/so_arm101_avoid/tasks/reach_avoid/assets/arm_lowpoly.usd",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import fast_simplification  # noqa: E402
from pxr import Usd, UsdGeom, Vt  # noqa: E402

stage = Usd.Stage.Open(args.src)
mesh_prim = next(p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
mesh = UsdGeom.Mesh(mesh_prim)

pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
assert np.all(counts == 3), "expected a triangulated mesh"
faces = idx.reshape(-1, 3)
print(f"[decimate] src: {len(pts)} pts, {len(faces)} tris", flush=True)

# weld duplicate vertices so quadric decimation is well-behaved
uniq, inv = np.unique(pts, axis=0, return_inverse=True)
faces_w = inv[faces]
print(f"[decimate] welded to {len(uniq)} unique pts", flush=True)

target_reduction = max(0.0, 1.0 - args.target_tris / len(faces_w))
new_pts, new_faces = fast_simplification.simplify(
    uniq.astype(np.float32), faces_w.astype(np.int32), target_reduction=target_reduction
)
print(f"[decimate] dst: {len(new_pts)} pts, {len(new_faces)} tris", flush=True)

# write geometry back; drop now-invalid normals/UVs so the renderer recomputes
mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(new_pts.astype(np.float32)))
mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(new_faces), 3, np.int32)))
mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(new_faces.reshape(-1).astype(np.int32)))
if mesh.GetNormalsAttr().HasAuthoredValue():
    mesh.GetNormalsAttr().Clear()
for pv in UsdGeom.PrimvarsAPI(mesh_prim).GetPrimvars():
    if pv.GetInterpolation() in ("faceVarying", "vertex"):
        pv.GetAttr().Clear()

stage.Export(args.dst)
print(f"[decimate] wrote {args.dst}", flush=True)
app.close()
