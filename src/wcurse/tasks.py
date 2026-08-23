from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Task:

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

    import numpy as np
    from datasets import load_dataset

    ds = load_dataset(task.hf_path, task.hf_config, split=task.split)
    available = np.setdiff1d(np.arange(len(ds)), np.asarray(exclude, dtype=int))
    if available.size < n:
        raise ValueError(f"{task.name}: only {available.size} unscored items available")
    picks = np.random.default_rng(seed).choice(available, size=n, replace=False)
    return [(ds[int(i)][task.text_field], int(ds[int(i)][task.label_field])) for i in picks]


def render(prompt: str, text: str, task: Task) -> str:
    body = prompt.replace("{text}", text) if "{text}" in prompt else f"{prompt}\n\n{text}"
    options = " / ".join(task.verbalizers)
    return f"{body}\n\nOptions: {options}\nAnswer:"
