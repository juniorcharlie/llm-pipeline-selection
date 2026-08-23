from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
import re
from pathlib import Path

from wcurse.tasks import TASKS, Task

REPO = Path(__file__).resolve().parents[1]
SEEDS_PATH = REPO / "data" / "prompts" / "seeds.json"

APE_TEMPLATE = """Here are labelled examples from a text classification task.

{demos}

Write a single-sentence instruction that would tell an assistant how to label examples \
like these. Reply with the instruction only, no preamble and no examples.

Instruction:"""

MUTATION_TEMPLATE = """Rewrite the instruction below so that it means the same thing but \
is worded differently. Vary the phrasing, register, or level of detail. Reply with the \
rewritten instruction only.

Instruction: {seed}

Rewritten instruction:"""


class StubGenerator:
    """Deterministic stand-in for a model, used to test the plumbing without a GPU.

    It produces syntactically distinct strings so that de-duplication and top-up logic can
    be exercised; it does not produce useful prompts and must never be used for a matrix
    that appears in the paper.
    """

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompts: list[str], max_tokens: int = 64) -> list[str]:
        out = []
        for p in prompts:
            self.calls += 1
            tag = re.sub(r"\W+", "-", p[-40:]).strip("-")
            out.append(f"[stub {self.calls}] instruction derived from {tag}")
        return out


class VLLMGenerator:
    """Sampling wrapper over vLLM, used for the real pools."""

    def __init__(self, model: str, temperature: float, seed: int, max_model_len: int) -> None:
        from vllm import LLM, SamplingParams

        self.llm = LLM(model=model, dtype="half", max_model_len=max_model_len, seed=seed)
        self.params = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=64, seed=seed)
        self._SamplingParams = SamplingParams

    def generate(self, prompts: list[str], max_tokens: int = 64) -> list[str]:
        params = self._SamplingParams(
            temperature=self.params.temperature,
            top_p=self.params.top_p,
            max_tokens=max_tokens,
            seed=self.params.seed,
        )
        outs = self.llm.generate(prompts, params)
        return [o.outputs[0].text for o in outs]


def is_near_duplicate(text: str, existing: set[str] | list[str], threshold: float = 0.95) -> bool:
    """Check if text is semantically/string-wise too similar to any existing item."""
    return any(SequenceMatcher(None, text, e).ratio() > threshold for e in existing)

def clean(raw: str) -> str:
    """Reduce a model completion to a single-line instruction.

    Generated candidates arrive with quotes, numbering, and occasional trailing chatter.
    Left uncleaned, that formatting noise becomes a confound: a candidate could win the
    selection because of a stray newline rather than because of what it says.
    """
    text = raw.strip().split("\n")[0].strip()
    text = re.sub(r'^["\'\u201c\u2018]|["\'\u201d\u2019]$', "", text).strip()
    text = re.sub(r"^(instruction|task|prompt)\s*[:\-]\s*", "", text, flags=re.I)
    text = re.sub(r"^\d+[\.\)]\s*", "", text)
    return " ".join(text.split())


def format_demos(demos: list[tuple[str, int]], task: Task) -> str:
    return "\n\n".join(
        f"Input: {text}\nLabel: {task.verbalizers[y]}" for text, y in demos
    )


def build_pool(
    task: Task,
    gen,
    n_ape: int,
    n_mutation: int,
    demos_per_prompt: int,
    item_seed: int,
    gen_seed: int,
    offline: bool,
) -> list[dict]:
    """Assemble the pool, de-duplicating as it goes.

    Duplicates are dropped rather than kept because two identical rows would give the
    correlation analysis a spurious eigenvalue of exactly zero and would let the same
    instruction win twice.
    """
    seeds: list[str] = json.loads(SEEDS_PATH.read_text())[task.name]
    pool = [
        {"text": s, "provenance": "seed", "parent": None, "index": i}
        for i, s in enumerate(seeds)
    ]
    seen = {s.lower() for s in seeds}

    if offline:
        return pool

    from wcurse.tasks import load_demonstrations, load_items

    _, _, scored_idx = load_items(task, seed=item_seed)

    # APE-style: one instruction per independent demonstration set.
    ape_prompts = []
    for j in range(n_ape):
        demos = load_demonstrations(
            task, exclude=scored_idx, n=demos_per_prompt, seed=gen_seed + 1000 + j
        )
        ape_prompts.append(APE_TEMPLATE.format(demos=format_demos(demos, task)))

    # Mutations: seeds cycled so each seed contributes a comparable number of children.
    parents = [seeds[j % len(seeds)] for j in range(n_mutation)]
    mut_prompts = [MUTATION_TEMPLATE.format(seed=p) for p in parents]

    raw = gen.generate(ape_prompts + mut_prompts)
    ape_raw, mut_raw = raw[:n_ape], raw[n_ape:]

    for j, r in enumerate(ape_raw):
        text = clean(r)
        if len(text) < 15 or is_near_duplicate(text.lower(), seen):
            continue
        seen.add(text.lower())
        pool.append({"text": text, "provenance": "ape", "parent": None, "index": len(pool)})

    for j, r in enumerate(mut_raw):
        text = clean(r)
        if len(text) < 15 or is_near_duplicate(text.lower(), seen):
            continue
        seen.add(text.lower())
        pool.append(
            {
                "text": text,
                "provenance": "mutation",
                "parent": seeds.index(parents[j]),
                "index": len(pool),
            }
        )

    return pool


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--backend", default="vllm", choices=("vllm", "stub"))
    ap.add_argument("--offline", action="store_true", help="emit the 10 manual seeds only")
    ap.add_argument("--n-ape", type=int, default=45)
    ap.add_argument("--n-mutation", type=int, default=45)
    ap.add_argument("--demos-per-prompt", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--item-seed", type=int, default=20260818)
    ap.add_argument("--gen-seed", type=int, default=7)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    task = TASKS[args.task]
    if args.offline:
        gen = None
    elif args.backend == "stub":
        gen = StubGenerator()
    else:
        gen = VLLMGenerator(args.model, args.temperature, args.gen_seed, args.max_model_len)

    pool = build_pool(
        task,
        gen,
        n_ape=args.n_ape,
        n_mutation=args.n_mutation,
        demos_per_prompt=args.demos_per_prompt,
        item_seed=args.item_seed,
        gen_seed=args.gen_seed,
        offline=args.offline,
    )

    out = Path(args.out or REPO / "data" / "prompts" / f"{args.task}_pool.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "task": task.name,
                "generator_model": None if args.offline else args.model,
                "backend": "offline" if args.offline else args.backend,
                "temperature": args.temperature,
                "gen_seed": args.gen_seed,
                "item_seed": args.item_seed,
                "candidates": pool,
            },
            indent=2,
        )
    )
    counts = {p: sum(c["provenance"] == p for c in pool) for p in ("seed", "ape", "mutation")}
    print(f"wrote {len(pool)} candidates to {out}")
    print(f"provenance counts: {counts}")
    if len(pool) < 100 and not args.offline:
        print(
            f"WARNING: pool is {len(pool)} < 100 after de-duplication; "
            "raise --temperature or --n-mutation and regenerate"
        )


if __name__ == "__main__":
    main()
