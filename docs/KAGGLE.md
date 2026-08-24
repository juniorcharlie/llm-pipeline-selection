# Building the matrices on a Kaggle T4

The compute-bound step, and the only one. Budget: 30 GPU-hr per week, free tier, marginal
cost zero. The plan needs roughly 180,000 single-token generations for the primary model
(100 candidates x 600 items x 3 tasks).

## Notebook setup

Enable the GPU accelerator (T4 x1) and internet, then:

```bash
!pip install -q vllm==0.6.\* datasets
!git clone https://github.com/juniorcharlie/llm-pipeline-selection.git
%cd llm-pipeline-selection
!pip install -q -e .
```

## Pilot first

Never commit the quota before measuring throughput. Ten candidates is enough to extrapolate:

```bash
!python scripts/generate\_candidates.py --task sst2 --model Qwen/Qwen2.5-1.5B-Instruct
!python scripts/build\_matrix.py --task sst2 --limit-candidates 10 --pilot
```

The pilot prints a projection in GPU-hours for one task and for three. If the full matrix
does not fit in the quota with slack, **cut tasks or models, never `K` or `M`** — both are
independent variables in the analysis, and shrinking either one shrinks the paper.

## The real run

```bash
!python scripts/build\_matrix.py --task sst2
!python scripts/build\_matrix.py --task sst2 --finalize
```

Checkpointing is per candidate: `checkpoints/sst2\_\_Qwen\_\_Qwen2.5-1.5B-Instruct/preds\_0007.npy`
and so on. Rerunning the identical command after a killed session skips whatever is already
on disk, so a dead kernel costs at most one candidate. The item sample and the dev/truth
split are written to `meta.json` on the first call and verified on every later one, so a
resumed run cannot silently score a different item sample than the rows already stored.

Commit the matrices and the pool files to the repo; they are small (a 100 x 600 uint8 matrix
compresses to a few kilobytes) and they are the artifact everything else depends on.

## Order of work

1. `sst2` on Qwen2.5-1.5B-Instruct — the primary matrix, and the one the pilot calibrates
2. `subj`, then `agnews` on the same model
3. Llama-3.2-3B-Instruct on whatever tasks the quota still allows
4. `trec` and Qwen2.5-7B-Instruct only if everything above is done

If fewer than two complete matrices exist by the Friday checkpoint, cut to two tasks and one
model and move on to the analysis. The analysis is where the contributions are.

**Status: steps 1-3 complete, step 4 not reached.** `sst2`, `subj`, and `agnews` are all built
on Qwen2.5-1.5B-Instruct; `sst2` is additionally built on Llama-3.2-3B-Instruct, the only task
the remaining quota reached on that model. `trec` and Qwen2.5-7B-Instruct remain unbuilt, for
quota reasons rather than any blocker in the method. The Friday-checkpoint contingency above
did not trigger — four matrices, not two, existed by then. Extending to either stretch item is
the same procedure above, one more `generate\_candidates.py` / `build\_matrix.py` pass.

## Sanity checks on a fresh matrix

`build\_matrix.py --finalize` prints the accuracy range, the count of unparsed generations,
and the dev/truth offset. Read all three before trusting the matrix:

* **Accuracy range.** A pool where every candidate lands within a point of every other has
no selection to study; a pool where half the candidates sit at chance suggests the
verbalizers or the render template are broken.
* **Unparsed count.** Should be exactly zero. `allowed\_token\_ids` makes anything else a bug.
* **Dev/truth offset.** A few tenths of a point is normal and is subtracted in the analysis.
Several points means the stratified item sample did not do its job.

Then run the analysis.

```bash
!python scripts/run\_analysis.py --matrices data/matrices/sst2\_\_\*.npz
```

