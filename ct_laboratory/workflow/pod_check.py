"""Pod health gate: every GPU usable, CPU not starved, kernel launches fast. Exit 3 if the host is not fit for launch-bound work.
Thresholds from axis03 (python loop 0.11 s, launch 6 us) vs a starved US-IL-1 host (1.82 s, 47-82 us)."""
import sys, time, torch
n = torch.cuda.device_count(); [torch.zeros(1, device=f"cuda:{i}") for i in range(n)]
t = time.time(); s = 0
for i in range(3_000_000): s += i
cpu = time.time() - t
x = torch.zeros(1, device="cuda"); torch.cuda.synchronize(); t = time.time()
for _ in range(2000): x += 1
torch.cuda.synchronize(); lat = (time.time() - t) / 2000 * 1e6
ok = cpu < 0.4 and lat < 20
print(f"pod check: {n} GPUs | python loop {cpu:.2f} s (axis03 0.11) | kernel launch {lat:.1f} us (axis03 6) -> {'OK' if ok else 'TOO SLOW'}")
sys.exit(0 if ok else 3)
