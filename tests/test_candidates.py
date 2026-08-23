import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_candidates import SEEDS_PATH, StubGenerator, build_pool, clean  # noqa: E402
from wcurse.tasks import TASKS, render  # noqa: E402


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('  "Classify the sentiment."  ', "Classify the sentiment."),
        ("Instruction: Label the review.", "Label the review."),
        ("1. Decide the topic.\nExtra chatter here", "Decide the topic."),
        ("Read   the\ttext  and   label it.", "Read the text and label it."),
    ],
)
def test_clean_strips_formatting_noise(raw, expected):
    assert clean(raw) == expected


def test_seed_file_has_ten_distinct_seeds_per_task():
    seeds = json.loads(SEEDS_PATH.read_text())
    for name in TASKS:
        assert name in seeds, name
        assert len(seeds[name]) == 10, name
        assert len(set(seeds[name])) == 10, name


def test_offline_pool_is_the_seed_stratum_only():
    pool = build_pool(
        TASKS["sst2"],
        gen=None,
        n_ape=45,
        n_mutation=45,
        demos_per_prompt=5,
        item_seed=1,
        gen_seed=1,
        offline=True,
    )
    assert len(pool) == 10
    assert {c["provenance"] for c in pool} == {"seed"}


def test_stub_generator_produces_distinct_strings():
    gen = StubGenerator()
    out = gen.generate(["prompt one", "prompt two", "prompt one"])
    assert len(set(out)) == 3


def test_render_includes_the_item_and_the_option_list():
    task = TASKS["agnews"]
    text = render("Classify the topic.", "Stocks tumbled today.", task)
    assert "Stocks tumbled today." in text
    assert "World / Sports / Business / Technology" in text
    assert text.rstrip().endswith("Answer:")


def test_render_substitutes_a_placeholder_when_present():
    task = TASKS["sst2"]
    text = render("Review: {text}\nWhat is the sentiment?", "a dull film", task)
    assert "Review: a dull film" in text
    assert "{text}" not in text
