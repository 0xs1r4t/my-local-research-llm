from retrieval.hybrid import _embed
import time

q = "what is non-photorealistic rendering"

t0 = time.perf_counter()
e1 = _embed(q)
print(f"first call:  {(time.perf_counter()-t0)*1000:.1f}ms")

t0 = time.perf_counter()
e2 = _embed(q)
print(f"second call: {(time.perf_counter()-t0)*1000:.1f}ms")  # should be ~0ms

print(f"same result: {e1 == e2}")
print(_embed.cache_info())  # shows hits/misses