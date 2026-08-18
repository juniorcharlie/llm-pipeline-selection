"""Task registry: datasets, verbalizers, and item sampling.

Deliberately the same task set EvoPrompt used, so measured bias can be placed next to
published gains without an apples-to-oranges caveat. All four tasks are single-token
classification, which is the property that makes the compute budget work: scoring a
candidate on an item costs one forward pass and one constrained token.

No dataset library is imported at module load, so the analysis half of the repo stays
installable without any of the GPU-side dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Task:
    """Everything needed to score a candidate prompt on a task.

    Attributes:
        name: short identifier used in filenames and figures.
        hf_path: Hugging Face dataset path.
        hf_config: dataset config name, if any.
        split: split to draw evaluation items from.
        text_field: column holding the input text.
        label_field: column holding the integer label.
        verbalizers: class index -> the word the model is asked to emit. Order must match
            the dataset's own label ordering.
        n_items: number of items to score, split evenly into dev and truth pools.
    """

    name: str
    hf_path: str
    hf_config: str | None
    split: str
    text_field: str
    label_field: str
    verbalizers: tuple[str, ...]
    n_items: int = 600
    aliases: dict[int, tuple[str, ...]] = field(default_factory=dict)

    @property
    def n_classes(self) -> int:
        return len(self.verbalizers)


TASKS: dict[str, Task] = {
    "sst2": Task(
        name="sst2",
        hf_path="glue",
        hf_config="sst2",
        split="validation",
        text_field="sentence",
        label_field="label",
        verbalizers=("negative", "positive"),
    ),
    "subj": Task(
        name="subj",
        hf_path="SetFit/subj",
        hf_config=None,
        split="test",
        text_field="text",
        label_field="label",
        verbalizers=("objective", "subjective"),
    ),
    "agnews": Task(
        name="agnews",
        hf_path="ag_news",
        hf_config=None,
        split="test",
        text_field="text",
        label_field="label",
        verbalizers=("World", "Sports", "Business", "Technology"),
    ),
    "trec": Task(
        name="trec",
        hf_path="trec",
        hf_config=None,
        split="test",
        text_field="text",
        label_field="coarse_label",
        verbalizers=(
            "Abbreviation",
            "Entity",
            "Description",
            "Person",
            "Location",
            "Number",
        ),
    ),
}


def load_items(task: Task, seed: int) -> tuple[list[str], list[int], list[int]]:
    """Draw a class-stratified sample of ``task.n_items`` evaluation items.

    Stratification matters more than it looks: with 600 items split into two pools of 300,
    an unstratified draw can leave the pools with visibly different class balance, which
    shows up later as a spurious dev/truth difficulty offset.

    Returns:
        ``(texts, labels, source_indices)``. The source indices are returned so candidate
        generation can be restricted to items that are never scored.
    """
    import numpy as np
    from datasets import load_dataset

    ds = load_dataset(task.hf_path, task.hf_config, split=task.split)
    labels = np.asarray(ds[task.label_field])
    rng = np.random.default_rng(seed)

    per_class = task.n_items // task.n_classes
    chosen: list[int] = []
    for c in range(task.n_classes):
        pool = np.flatnonzero(labels == c)
        if pool.size < per_class:
            raise ValueError(
                f"{task.name}: class {c} has only {pool.size} items, need {per_class}"
            )
        chosen.extend(rng.choice(pool, size=per_class, replace=False).tolist())

    order = rng.permutation(len(chosen))
    idx = [int(chosen[i]) for i in order]
    texts = [ds[i][task.text_field] for i in idx]
    ys = [int(ds[i][task.label_field]) for i in idx]
    return texts, ys, idx


def load_demonstrations(
    task: Task, exclude: list[int], n: int, seed: int
) -> list[tuple[str, int]]:
    """Sample labelled demonstrations for APE-style candidate generation.

    Drawn strictly from items outside ``exclude``, that is, outside the 600 scored items. A
    candidate written from an item that is later scored would leak the truth pool into the
    candidate pool, and the leak would look like a real improvement.
    """
    import numpy as np
    from datasets import load_dataset

    ds = load_dataset(task.hf_path, task.hf_config, split=task.split)
    available = np.setdiff1d(np.arange(len(ds)), np.asarray(exclude, dtype=int))
    if available.size < n:
        raise ValueError(f"{task.name}: only {available.size} unscored items available")
    picks = np.random.default_rng(seed).choice(available, size=n, replace=False)
    return [(ds[int(i)][task.text_field], int(ds[int(i)][task.label_field])) for i in picks]


def render(prompt: str, text: str, task: Task) -> str:
    """Fill a candidate prompt template with one item.

    Candidates are stored as instructions; ``{text}`` is substituted if present and the
    item is otherwise appended, so hand-written and model-generated instructions can share
    one code path. The trailing ``Answer:`` is fixed across candidates on purpose: varying
    the answer cue would confound instruction quality with output-format luck.
    """
    body = prompt.replace("{text}", text) if "{text}" in prompt else f"{prompt}\n\n{text}"
    options = " / ".join(task.verbalizers)
    return f"{body}\n\nOptions: {options}\nAnswer:"
