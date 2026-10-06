"""MultiDeviceProjector3D == single VoxelProjector3D (forward, adjoint, autograd).

Runs on whatever is available: N CUDA devices (default all), or a single device split
into several parts (same device repeated) - the split/concatenate logic is identical.
    python python_examples/test_multi_device_projector_3d.py [n_parts]
"""
import sys, math, torch
from ct_laboratory.tomography import VoxelProjector3D, split_voxel_projector, make_views

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
backend = "cuda" if dev == "cuda" else "torch"
n_gpu = torch.cuda.device_count()
n_parts = int(sys.argv[1]) if len(sys.argv) > 1 else max(n_gpu, 3)
devices = [f"cuda:{k % n_gpu}" if n_gpu else "cpu" for k in range(n_parts)]

# small ring: 24 sources x 3 panels, 32x8 pixels
n_src, n_mod, nu, nv, R_src, R_det = 24, 3, 32, 8, 480.0, 400.0
S, C, U, V = [], [], [], []
for s in range(n_src):
    a = 2 * math.pi * s / n_src
    for m in (-1, 0, 1):
        g = a + math.pi + 0.25 * m
        S.append([R_src * math.cos(a), R_src * math.sin(a), 0.0])
        C.append([R_det * math.cos(g), R_det * math.sin(g), 5.0])
        U.append([-math.sin(g), math.cos(g), 0.0]); V.append([0.0, 0.0, 1.0])
S, C, U, V = (torch.tensor(t) for t in (S, C, U, V))
idx = torch.arange(S.shape[0])
views, _ = make_views(S, C, U, V, 4.0, 4.0, nu, nv, pairs=torch.stack([idx, idx], 1), device=dev)
n = 48
M = torch.diag(torch.tensor([4.0, 4.0, 4.0])).to(dev); b = torch.tensor([-96.0, -96.0, -16.0]).to(dev)
valid = (torch.rand(views.shape[0] * nu * nv) > 0.05).to(dev)

kw = dict(backend=backend, cache="column" if backend == "cuda" else "none")
A1 = VoxelProjector3D(n, n, 8, M, b, views, valid=valid, device=dev, **kw)
AN = split_voxel_projector(n, n, 8, M, b, views, valid=valid, devices=devices, output_device=dev, **kw)
print(AN)
x = torch.rand(n, n, 8, device=dev); y = torch.rand(A1.n_ray, device=dev)
ok = True
def check(name, a, b, tol=1e-5):
    global ok
    e = float((a - b).norm() / b.norm()); ok &= e < tol
    print(f"{'PASS' if e < tol else 'FAIL'} {name}: rel err {e:.2e}")
check("forward", AN.forward_project(x), A1.forward_project(x))
check("back", AN.back_project(y), A1.back_project(y))
check("batched forward", AN.forward_project(torch.stack([x, 2 * x])), A1.forward_project(torch.stack([x, 2 * x])))
lhs = float((AN.forward_project(x) * y).sum()); rhs = float((x * AN.back_project(y)).sum())
check("adjoint <Ax,y>=<x,A'y>", torch.tensor(lhs), torch.tensor(rhs), 1e-4)
xr = x.clone().requires_grad_(); (AN(xr) * y).sum().backward()
check("autograd grad", xr.grad, A1.back_project(y))
print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
