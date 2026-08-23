from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

from wcurse.tasks import TASKS, Task, render

REPO = Path(__file__).resolve().parents[1]
SEEDS_PATH = REPO / "data" / "prompts" / "seeds.json"

CROSSOVER_TEMPLATE = """You are improving an instruction for a text classification task.

Parent instruction A: {a}

Parent instruction B: {b}

Write one new instruction that combines the best ideas of A and B, then reword it so it \
is not a copy of either. Reply with the new instruction only, no preamble.

New instruction:"""


def slug(model: str) -> str:
    return model.replace("/", "__")


def clean(raw: str) -> str:
    text = raw.strip().split("\n")[0].strip()
    text = re.sub(r'^["\'\u201c\u2018]|["\'\u201d\u2019]$', "", text).strip()
    text = re.sub(r"^(instruction|task|prompt)\s*[:\-]\s*", "", text, flags=re.I)
    text = re.sub(r"^\d+[\.\)]\s*", "", text)
    return " ".join(text.split())


# --------------------------------------------------------------------------------------
# Backends: a stub for offline plumbing checks, vLLM for the real thing. Both expose
# .score(prompt_text, texts, labels) -> accuracy and .crossover(a, b) -> new prompt text,
# so the GA loop below never has to know which one it's talking to.
# --------------------------------------------------------------------------------------


class StubBackend:

    def score(self, prompt_text: str, texts: list[str], labels: list[int]) -> float:
        h = int(hashlib.sha256(prompt_text.encode()).hexdigest(), 16)
        return 0.5 + 0.4 * ((h % 1000) / 1000.0 - 0.5)

    def crossover(self, a: str, b: str) -> str:
        tag = re.sub(r"\W+", "-", (a + b)[-30:]).strip("-")
        return f"Combined instruction derived from {tag}: classify the input."


class VLLMBackend:

    def __init__(self, model: str, task: Task, seed: int, max_model_len: int) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.task = task
        self.llm = LLM(model=model, dtype="half", max_model_len=max_model_len, seed=seed)
        tokenizer = AutoTokenizer.from_pretrained(model)
        self.allowed_ids = []
        for word in task.verbalizers:
            toks = tokenizer.encode(f" {word}", add_special_tokens=False)
            if not toks:
                raise ValueError(f"verbalizer {word!r} tokenizes to nothing")
            self.allowed_ids.append(int(toks[0]))
        if len(set(self.allowed_ids)) != len(self.allowed_ids):
            raise ValueError(f"{task.name}: verbalizers share a first token")
        self.id_to_class = {tid: c for c, tid in enumerate(self.allowed_ids)}
        self._score_params = SamplingParams(
            temperature=0.0, max_tokens=1, logprobs=len(self.allowed_ids),
            allowed_token_ids=self.allowed_ids,
        )
        self._xover_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=64, seed=seed)

    def score(self, prompt_text: str, texts: list[str], labels: list[int]) -> float:
        prompts = [render(prompt_text, t, self.task) for t in texts]
        outs = self.llm.generate(prompts, self._score_params)
        preds = np.array(
            [self.id_to_class.get(int(o.outputs[0].token_ids[0]), -1) for o in outs]
        )
        return float((preds == np.asarray(labels)).mean())

    def crossover(self, a: str, b: str) -> str:
        prompt = CROSSOVER_TEMPLATE.format(a=a, b=b)
        out = self.llm.generate([prompt], self._xover_params)
        return clean(out[0].outputs[0].text)


# --------------------------------------------------------------------------------------
# The GA loop itself.
# --------------------------------------------------------------------------------------


def load_run_meta(task: Task, model: str, ckpt_root: Path) -> dict:
    matrix_path = REPO / "data" / "matrices" / f"{task.name}__{slug(model)}.npz"
    ckpt_meta_path = ckpt_root / f"{task.name}__{slug(model)}" / "meta.json"

    if not matrix_path.exists():
        raise SystemExit(
            f"no matrix at {matrix_path}; build it first so the replay uses the exact "
            "same dev/truth split as the rest of the paper"
        )
    from wcurse.matrix import CorrectnessMatrix

    cm = CorrectnessMatrix.load(matrix_path)

    if ckpt_meta_path.exists():
        meta = json.loads(ckpt_meta_path.read_text())
        print(f"using checkpoint metadata at {ckpt_meta_path}")
        return meta

    from wcurse.tasks import load_items

    print(f"no checkpoint at {ckpt_meta_path}; re-deriving items from item_seed={cm.meta['item_seed']}")
    texts, labels, source_idx = load_items(task, seed=cm.meta["item_seed"])
    if source_idx != cm.meta["source_idx"] or labels != cm.meta["labels"]:
        raise SystemExit(
            "re-derived items don't match the matrix's stored source_idx/labels -- "
            "dataset version or item_seed has drifted since the matrix was built; "
            "do not proceed without resolving this"
        )
    return {
        "task": task.name, "model": model,
        "item_seed": cm.meta["item_seed"], "split_seed": cm.meta["split_seed"],
        "texts": texts, "labels": labels, "source_idx": source_idx,
        "dev_idx": cm.dev_idx.tolist(), "truth_idx": cm.truth_idx.tolist(),
        "verbalizers": cm.meta["verbalizers"],
    }


def fitness_batch(meta: dict, m: int, seed: int) -> tuple[list[str], list[int]]:
    rng = np.random.default_rng(seed)
    dev_idx = np.asarray(meta["dev_idx"])
    cols = rng.choice(dev_idx, size=m, replace=False)
    texts = [meta["texts"][i] for i in cols]
    labels = [meta["labels"][i] for i in cols]
    return texts, labels


def roulette_select(population: list[dict], rng: np.random.Generator, k: int = 2) -> list[dict]:
    fit = np.array([p["fitness"] for p in population])
    weights = fit - fit.min() + 0.05
    probs = weights / weights.sum()
    idx = rng.choice(len(population), size=k, replace=False, p=probs)
    return [population[i] for i in idx]


def run_trajectory(
    task: Task,
    backend,
    meta: dict,
    population_size: int,
    generations: int,
    m: int,
    seed: int,
    out_path: Path,
) -> dict:
    rng = np.random.default_rng(seed)
    fit_texts, fit_labels = fitness_batch(meta, m, seed=seed)

    state = _load_checkpoint(out_path)
    if state is None:
        seeds: list[str] = json.loads(SEEDS_PATH.read_text())[task.name][:population_size]
        print(f"scoring {len(seeds)} seed prompts on the fixed m={m} fitness batch")
        population = [{"text": s, "fitness": backend.score(s, fit_texts, fit_labels)} for s in seeds]
        touched = list(population)
        history = []
        start_gen = 1
        state = {
            "task": task.name, "seed": seed, "m": m,
            "population_size": population_size, "generations": generations,
            "population": population, "touched": touched, "history": history,
        }
        _save_checkpoint(out_path, state)
    else:
        population, touched, history = state["population"], state["touched"], state["history"]
        start_gen = len(history) + 1
        print(f"resuming from generation {start_gen} ({len(touched)} prompts touched so far)")

    for g in range(start_gen, generations + 1):
        t0 = time.time()
        parents = roulette_select(population, rng)
        offspring_text = backend.crossover(parents[0]["text"], parents[1]["text"])
        if len(offspring_text) < 15 or offspring_text.lower() in {t["text"].lower() for t in touched}:
            offspring_text = f"{offspring_text} (variant {g})"  # break the collision, keep the trajectory moving
        offspring_fitness = backend.score(offspring_text, fit_texts, fit_labels)
        offspring = {"text": offspring_text, "fitness": offspring_fitness}
        touched.append(offspring)

        worst_idx = int(np.argmin([p["fitness"] for p in population]))
        if offspring_fitness > population[worst_idx]["fitness"]:
            population[worst_idx] = offspring

        best_so_far = max(touched, key=lambda p: p["fitness"])
        history.append(
            {
                "generation": g,
                "n_touched": len(touched),
                "offspring_fitness": offspring_fitness,
                "best_fitness_so_far": best_so_far["fitness"],
                "best_text_so_far": best_so_far["text"],
            }
        )
        _save_checkpoint(out_path, state)
        print(
            f"  gen {g:2d}/{generations}  offspring={offspring_fitness:.3f}  "
            f"best={best_so_far['fitness']:.3f}  touched={len(touched)}  "
            f"({time.time() - t0:.1f}s)"
        )

    final_best = max(touched, key=lambda p: p["fitness"])
    truth_idx = np.asarray(meta["truth_idx"])
    truth_texts = [meta["texts"][i] for i in truth_idx]
    truth_labels = [meta["labels"][i] for i in truth_idx]
    true_score = backend.score(final_best["text"], truth_texts, truth_labels)

    state["final_best"] = final_best
    state["true_score"] = true_score
    state["bias_live"] = final_best["fitness"] - true_score
    state["n_live"] = len(touched)
    _save_checkpoint(out_path, state)
    return state


def _save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def _load_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    state = json.loads(path.read_text())
    if "final_best" in state:
        print(f"{path} already has a finished trajectory; delete it to rerun from scratch")
        raise SystemExit(0)
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--backend", default="vllm", choices=("vllm", "stub"))
    ap.add_argument("--population", type=int, default=10)
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--m", type=int, default=100, help="fixed dev fitness-batch size")
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--ckpt-root", default=str(REPO / "checkpoints"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    task = TASKS[args.task]
    meta = load_run_meta(task, args.model, Path(args.ckpt_root))
    out_path = Path(
        args.out or REPO / "results" / "evoprompt_replay" / f"{args.task}.json"
    )

    backend = StubBackend() if args.backend == "stub" else VLLMBackend(
        args.model, task, args.seed, args.max_model_len
    )

    state = run_trajectory(
        task, backend, meta,
        population_size=args.population, generations=args.generations, m=args.m,
        seed=args.seed, out_path=out_path,
    )

    print(
        f"\nfinal: N_live={state['n_live']}  m={state['m']}  "
        f"dev={state['final_best']['fitness']:.4f}  true={state['true_score']:.4f}  "
        f"bias_live={state['bias_live']:+.4f}"
    )
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
