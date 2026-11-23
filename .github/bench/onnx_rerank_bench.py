"""Reranker backends on an x86 CPU limited to 2 threads (as Cloud Run's 2 vCPUs): PyTorch vs the
model's published ONNX files. Same 50-candidate pools the service reranks, for the 48
ground-truth questions and the 50 attack questions. Reports latency and whether each backend
gives the same top-5 (order included) as PyTorch."""
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
import torch  # noqa: E402

torch.set_num_threads(2)

from sentence_transformers import CrossEncoder  # noqa: E402

from sentinel.config import ADVERSARIAL_DIR, ADVERSARIAL_INDEX_DIR, DATA_DIR, settings  # noqa: E402
from sentinel.retrieve import (  # noqa: E402
    DEFAULT_ACCESS,
    RERANK_POOL_SIZE,
    Access,
    _load_index,
    dense_search,
    rrf_fuse,
    sparse_search,
)

MODEL = settings.reranker_model
THREADS = 2
REPEATS = int(os.environ.get("BENCH_REPEATS", "3"))
K = settings.rerank_top_k


def pool(question, index_dir, access):
    dense = dense_search(question, index_dir=index_dir, access=access)
    sparse = sparse_search(question, index_dir=index_dir, access=access)
    fused = rrf_fuse([[c for c, _ in dense], [c for c, _ in sparse]])
    ids = sorted(fused, key=lambda c: fused[c], reverse=True)[:RERANK_POOL_SIZE]
    idx = _load_index(index_dir)
    return ids, [[question, idx.id2chunk[c].text] for c in ids]


gt = [json.loads(line)["question"] for line in (DATA_DIR / "ground_truth.jsonl").open()]
attacks = [json.loads(line)["question"] for line in (ADVERSARIAL_DIR / "questions.jsonl").open()]
pools = [("gt", *pool(q, None, DEFAULT_ACCESS)) for q in gt]
pools += [("attack", *pool(q, ADVERSARIAL_INDEX_DIR, Access())) for q in attacks]
if os.environ.get("BENCH_SMOKE"):
    pools = pools[:2] + pools[-2:]
print(f"{len(pools)} pools, sizes {sorted({len(p[1]) for p in pools})}", flush=True)


def onnx(file_name):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    so.inter_op_num_threads = 1
    return CrossEncoder(MODEL, backend="onnx", model_kwargs={
        "file_name": file_name, "provider": "CPUExecutionProvider", "session_options": so})


VARIANTS = {
    "torch": lambda: CrossEncoder(MODEL, device="cpu"),
    "onnx fp32": lambda: onnx("onnx/model.onnx"),
    "onnx fp32 O3": lambda: onnx("onnx/model_O3.onnx"),
    "onnx int8 avx2": lambda: onnx("onnx/model_quint8_avx2.onnx"),
    "onnx int8 avx512": lambda: onnx("onnx/model_qint8_avx512.onnx"),
    "onnx int8 avx512_vnni": lambda: onnx("onnx/model_qint8_avx512_vnni.onnx"),
}


def top(ids, scores):
    ranked = sorted(zip(ids, scores, strict=True), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked][:K]


results, reference = {}, None
for name, make in VARIANTS.items():
    try:
        model = make()
    except Exception as exc:  # noqa: BLE001
        results[name] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        print(name, "FAILED", results[name]["error"], flush=True)
        continue
    for _, _, pairs in pools[:3]:
        model.predict(pairs)
    times, scores = [], []
    for rep in range(REPEATS):
        for _, _, pairs in pools:
            t0 = time.perf_counter()
            s = model.predict(pairs)
            times.append((time.perf_counter() - t0) * 1000.0)
            if rep == 0:
                scores.append([float(x) for x in s])
    times.sort()
    row = {"p50_ms": round(times[len(times) // 2], 1),
           "p95_ms": round(times[int(len(times) * 0.95)], 1)}
    if reference is None:
        reference = scores
    else:
        same_order = same_set = 0
        diffs = []
        by_kind = {"gt": [0, 0], "attack": [0, 0]}
        for (kind, ids, _), ref, got in zip(pools, reference, scores, strict=True):
            a, b = top(ids, ref), top(ids, got)
            same_order += a == b
            same_set += set(a) == set(b)
            by_kind[kind][0] += a == b
            by_kind[kind][1] += 1
            diffs.append(max(abs(x - y) for x, y in zip(ref, got, strict=True)))
        row.update(same_top5_order=f"{same_order}/{len(pools)}",
                   same_top5_set=f"{same_set}/{len(pools)}",
                   same_order_gt="{}/{}".format(*by_kind["gt"]),
                   same_order_attack="{}/{}".format(*by_kind["attack"]),
                   max_score_diff=round(max(diffs), 4))
    results[name] = row
    print(name, row, flush=True)

cpu = {}
for line in open("/proc/cpuinfo"):
    if line.startswith("model name") and "model" not in cpu:
        cpu["model"] = line.split(":", 1)[1].strip()
    if line.startswith("flags") and "flags" not in cpu:
        flags = line.split(":", 1)[1].split()
        wanted = ("avx2", "avx512f", "avx512_vnni", "avx_vnni", "amx_int8")
        cpu["flags"] = [f for f in wanted if f in flags]
out = {"cpu": cpu, "nproc": os.cpu_count(), "threads": THREADS, "repeats": REPEATS,
       "pools": len(pools), "pool_size": RERANK_POOL_SIZE, "results": results}
print(json.dumps(out, indent=1))
json.dump(out, open("onnx_bench.json", "w"), indent=1)
