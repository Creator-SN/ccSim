# %%
import torch
from torch.utils import benchmark

torch.cuda.empty_cache()
tps = {"FP16": torch.float16, "FP32": torch.float32, "FP64": torch.float64}

for name in tps:
    typ = tps[name]
    n = 1024 * 16
    a = torch.randn(n, n).type(typ).cuda()
    b = torch.randn(n, n).type(typ).cuda()

    t = benchmark.Timer(stmt="a @ b", globals={"a": a, "b": b})

    x = t.timeit(50)
    print(f"{name} : {round(2 * n**3 / x.median / 1e12,2)}")

# %%
