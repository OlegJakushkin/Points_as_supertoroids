"""Public-library reconstruction baselines, run honestly (not reimplemented):
  - SPSR  : Screened Poisson [Kazhdan & Hoppe 2013] via Open3D [Zhou et al. 2018]
  - BPA   : Ball-Pivoting    [Bernardini et al. 1999] via Open3D
  - GWN   : (Fast) Generalized Winding Number [Barill et al. 2018; Jacobson et al. 2013] via libigl
Each returns (verts, faces, seconds).  Missing libraries -> the method is skipped by the caller."""
import time, numpy as np
from skimage import measure

BOUND = 1.1


def _o3d_pcd(P, N):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(P, np.float64))
    pcd.normals = o3d.utility.Vector3dVector(np.asarray(N, np.float64))
    return pcd, o3d


def _ml_meshset(P, N):
    import pymeshlab as ml
    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(vertex_matrix=np.asarray(P, np.float64), v_normals_matrix=np.asarray(N, np.float64)))
    return ms


def recon_spsr(P, N, depth=8, density_q=0.04):
    # open3d has no cp313 wheels; pymeshlab wraps the SAME original screened-Poisson code (Kazhdan 2013),
    # so fall back to it rather than skipping the method.  (No density-trim there -> classic untrimmed SPSR.)
    try:
        pcd, o3d = _o3d_pcd(P, N)
    except ImportError:
        ms = _ml_meshset(P, N)
        t = time.time()
        ms.generate_surface_reconstruction_screened_poisson(depth=depth)
        dt = time.time() - t
        m = ms.current_mesh()
        return np.asarray(m.vertex_matrix()), np.asarray(m.face_matrix()), dt
    t = time.time()
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, density_q))   # trim the Poisson bubble
    dt = time.time() - t
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles), dt


def recon_bpa(P, N):
    try:
        pcd, o3d = _o3d_pcd(P, N)
    except ImportError:
        ms = _ml_meshset(P, N)                                   # MeshLab ball-pivoting (Bernardini 1999)
        t = time.time()
        ms.generate_surface_reconstruction_ball_pivoting()
        dt = time.time() - t
        m = ms.current_mesh()
        return np.asarray(m.vertex_matrix()), np.asarray(m.face_matrix()), dt
    d = np.asarray(pcd.compute_nearest_neighbor_distance()); r = float(np.mean(d))
    t = time.time()
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([r * 1.5, r * 3.0, r * 6.0]))
    dt = time.time() - t
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles), dt


def recon_gwn(P, N, res=96):
    import igl
    P = np.asarray(P, np.float64); N = np.asarray(N, np.float64)
    lin = np.linspace(-BOUND, BOUND, res)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    Q = np.stack([X, Y, Z], -1).reshape(-1, 3).astype(np.float64)
    t = time.time()
    try:
        w = igl.fast_winding_number(P, N, Q)                       # (P, N, Q)
    except Exception:
        A = np.ones(len(P))
        w = igl.fast_winding_number(P, N, A, Q)                    # (P, N, A, Q)
    dt = time.time() - t
    g = w.reshape(res, res, res) - 0.5                             # surface at winding number 0.5
    if not (g.min() < 0 < g.max()):
        return None, None, dt
    v, f, _, _ = measure.marching_cubes(g, 0.0)
    return v / (res - 1) * (2 * BOUND) - BOUND, f, dt


def _pymeshlab_recon(P, N, which):
    import pymeshlab as ml
    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(vertex_matrix=np.asarray(P, np.float64), v_normals_matrix=np.asarray(N, np.float64)))
    t = time.time()
    gen = ms.generate_marching_cubes_apss if which == "APSS" else ms.generate_marching_cubes_rimls
    try:    gen(resolution=128)
    except Exception:
        try: gen()
        except Exception as e: raise e
    dt = time.time() - t
    m = ms.current_mesh()
    return np.asarray(m.vertex_matrix()), np.asarray(m.face_matrix()), dt


def recon_apss(P, N):  return _pymeshlab_recon(P, N, "APSS")    # Algebraic Point Set Surfaces (Guennebaud&Gross 2007)
def recon_rimls(P, N): return _pymeshlab_recon(P, N, "RIMLS")   # Robust Implicit MLS (Oztireli et al. 2009)


# name -> (callable, needs-import-module) so the driver can probe availability.
# NOTE: GWN (point-cloud fast winding number, Barill 2018) is NOT included -- the libigl python binding here
# only exposes the *mesh* winding number fast_winding_number(V,F,Q), not the point-cloud variant, and we do not
# reimplement methods.  Research methods without a drop-in public Python library (SHM/signed-heat, SSPD, SHC,
# NN-VIPSS, regularized winding number) are likewise omitted rather than faked.
METHODS = {
    "SPSR":  (recon_spsr,  ("open3d", "pymeshlab")),   # either library provides the algorithm
    "BPA":   (recon_bpa,   ("open3d", "pymeshlab")),
    "APSS":  (recon_apss,  ("pymeshlab",)),
    "RIMLS": (recon_rimls, ("pymeshlab",)),
}


def available():
    import importlib
    out = {}
    for name, (fn, mods) in METHODS.items():
        for mod in (mods if isinstance(mods, tuple) else (mods,)):
            try:
                importlib.import_module(mod); out[name] = fn; break
            except Exception:
                pass
    return out
