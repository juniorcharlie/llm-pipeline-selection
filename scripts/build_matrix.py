from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from wcurse.matrix import CorrectnessMatrix, make_splits
from wcurse.tasks import TASKS, Task, load_items, render

REPO = Path(__file__).resolve().parents[1]


def slug(model: str) -> str:
    return model.replace("/", "__")


def verbalizer_token_ids(tokenizer, task: Task) -> list[int]:
    ids = []
    for word in task.verbalizers:
        toks = tokenizer.encode(f" {word}", add_special_tokens=False)
        if not toks:
            raise ValueError(f"verbalizer {word!r} tokenizes to nothing")
        ids.append(int(toks[0]))
    if len(set(ids)) != len(ids):
        raise ValueError(
            f"{task.name}: verbalizers {task.verbalizers} share a first token "
            f"(ids={ids}); pick different label words"
        )
    return ids


def prepare_run(task: Task, model: str, item_seed: int, split_seed: int, ckpt_dir: Path) -> dict:
    meta_path = ckpt_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["task"] != task.name or meta["model"] != model:
            raise ValueError(f"checkpoint at {ckpt_dir} belongs to a different run: {meta}")
        return meta

    texts, labels, source_idx = load_items(task, seed=item_seed)
    n_dev = n_truth = task.n_items // 2
    dev_idx, truth_idx = make_splits(len(texts), n_dev, n_truth, seed=split_seed)
    meta = {
        "task": task.name,
        "model": model,
        "item_seed": item_seed,
        "split_seed": split_seed,
        "texts": texts,
        "labels": labels,
        "source_idx": source_idx,
        "dev_idx": dev_idx.tolist(),
        "truth_idx": truth_idx.tolist(),
        "verbalizers": list(task.verbalizers),
    }
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta))
    return meta


def score_candidates(
    llm,
    params_factory,
    pool: list[dict],
    meta: dict,
    task: Task,
    ckpt_dir: Path,
    allowed_ids: list[int],
    pilot: bool,
) -> None:
    
    texts, labels = meta["texts"], np.asarray(meta["labels"])
    params = params_factory(allowed_ids)
    id_to_class = {tid: c for c, tid in enumerate(allowed_ids)}

    t_start = time.time()
    n_done = 0
    for k, cand in enumerate(pool):
        out_path = ckpt_dir / f"preds_{k:04d}.npy"
        if out_path.exists():
            continue
        prompts = [render(cand["text"], t, task) for t in texts]
        t0 = time.time()
        outs = llm.generate(prompts, params)
        preds = np.array(
            [id_to_class.get(int(o.outputs[0].token_ids[0]), -1) for o in outs], dtype=np.int8
        )
        np.save(out_path, preds)
        n_done += 1
        dt = time.time() - t0
        acc = float((preds == labels).mean())
        print(
            f"[{k + 1}/{len(pool)}] {cand['provenance']:<8} acc={acc:.3f} "
            f"{dt:.1f}s ({len(prompts) / dt:.0f} items/s)",
            flush=True,
        )

    if pilot and n_done:
        elapsed = time.time() - t_start
        per_cand = elapsed / n_done
        print(
            f"\npilot: {n_done} candidates x {len(texts)} items in {elapsed / 60:.1f} min "
            f"({per_cand:.1f}s per candidate)\n"
            f"projection: 100 candidates = {100 * per_cand / 3600:.2f} GPU-hr per task, "
            f"3 tasks = {300 * per_cand / 3600:.2f} GPU-hr"
        )


def finalize(pool: list[dict], meta: dict, ckpt_dir: Path, out_path: Path) -> CorrectnessMatrix:
    labels = np.asarray(meta["labels"])
    missing = [k for k in range(len(pool)) if not (ckpt_dir / f"preds_{k:04d}.npy").exists()]
    if missing:
        raise SystemExit(f"cannot finalize: {len(missing)} candidates unscored, e.g. {missing[:5]}")

    preds = np.stack([np.load(ckpt_dir / f"preds_{k:04d}.npy") for k in range(len(pool))])
    n_unparsed = int((preds < 0).sum())
    C = (preds == labels[None, :]).astype(np.uint8)

    cm = CorrectnessMatrix(
        C=C,
        dev_idx=np.asarray(meta["dev_idx"]),
        truth_idx=np.asarray(meta["truth_idx"]),
        provenance=np.array([c["provenance"] for c in pool]),
        task=meta["task"],
        model=meta["model"],
        meta={
            "item_seed": meta["item_seed"],
            "split_seed": meta["split_seed"],
            "verbalizers": meta["verbalizers"],
            "source_idx": meta["source_idx"],
            "labels": meta["labels"],
            "n_unparsed": n_unparsed,
            "pool_sha256": hashlib.sha256(
                json.dumps([c["text"] for c in pool]).encode()
            ).hexdigest(),
        },
    )
    cm.save(out_path)
    acc = C.mean(axis=1)
    print(
        f"wrote {out_path}\n"
        f"  K={cm.K} M={cm.M} unparsed={n_unparsed}\n"
        f"  accuracy: min={acc.min():.3f} median={np.median(acc):.3f} max={acc.max():.3f}\n"
        f"  dev/truth offset={cm.dev().mean() - cm.truth().mean():+.4f}"
    )
    return cm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--pool", default=None, help="defaults to data/prompts/{task}_pool.json")
    ap.add_argument("--out", default=None, help="defaults to data/matrices/{task}__{model}.npz")
    ap.add_argument("--ckpt-root", default=str(REPO / "checkpoints"))
    ap.add_argument("--limit-candidates", type=int, default=None)
    ap.add_argument("--item-seed", type=int, default=20260818)
    ap.add_argument("--split-seed", type=int, default=20260819)
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--pilot", action="store_true", help="print a throughput extrapolation")
    ap.add_argument("--finalize", action="store_true", help="assemble checkpoints and exit")
    args = ap.parse_args()

    task = TASKS[args.task]
    pool_path = Path(args.pool or REPO / "data" / "prompts" / f"{args.task}_pool.json")
    pool = json.loads(pool_path.read_text())["candidates"]
    if args.limit_candidates:
        pool = pool[: args.limit_candidates]

    ckpt_dir = Path(args.ckpt_root) / f"{task.name}__{slug(args.model)}"
    out_path = Path(args.out or REPO / "data" / "matrices" / f"{task.name}__{slug(args.model)}.npz")

    meta = prepare_run(task, args.model, args.item_seed, args.split_seed, ckpt_dir)

    if args.finalize:
        finalize(pool, meta, ckpt_dir, out_path)
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    allowed_ids = verbalizer_token_ids(tokenizer, task)
    print(f"verbalizer token ids: {dict(zip(task.verbalizers, allowed_ids))}")

    llm = LLM(
        model=args.model,
        dtype="half",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=0,
    )

    def params_factory(ids: list[int]):
        return SamplingParams(
            temperature=0.0, max_tokens=1, logprobs=len(ids), allowed_token_ids=ids
        )

    score_candidates(
        llm, params_factory, pool, meta, task, ckpt_dir, allowed_ids, pilot=args.pilot
    )
    print(f"\ncheckpoints in {ckpt_dir}; rerun with --finalize to assemble the matrix")


if __name__ == "__main__":
    main()
