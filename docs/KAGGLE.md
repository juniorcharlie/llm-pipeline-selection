# Building the matrices on a Kaggle T4

The compute-bound step, and the only one. Budget: 30 GPU-hr per week, free tier, marginal
cost zero. The plan needs roughly 180,000 single-token generations for the primary model
(100 candidates x 600 items x 3 tasks).

## Notebook setup

Enable the GPU accelerator (T4 x1) and internet, then:

```bash
!pip install -q vllm==0.6.* datasets
!git clone https://github.com/juniorcharlie/llm-pipeline-selection.git
%cd llm-pipeline-selection
!pip install -q -e .
```

## Pilot first

Never commit the quota before measuring throughput. Ten candidates is enough to extrapolate:

```bash
!python scripts/generate_candidates.py --task sst2 --model Qwen/Qwen2.5-1.5B-Instruct
!python scripts/build_matrix.py --task sst2 --limit-candidates 10 --pilot
```

The pilot prints a projection in GPU-hours for one task and for three. If the full matrix
does not fit in the quota with slack, **cut tasks or models, never `K` or `M`** — both are
independent variables in the analysis, and shrinking either one shrinks the paper.

## The real run

```bash
!python scripts/build_matrix.py --task sst2
!python scripts/build_matrix.py --task sst2 --finalize
```

Checkpointing is per candidate: `checkpoints/sst2__Qwen__Qwen2.5-1.5B-Instruct/preds_0007.npy`
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

## Sanity checks on a fresh matrix

`build_matrix.py --finalize` prints the accuracy range, the count of unparsed generations,
and the dev/truth offset. Read all three before trusting the matrix:

- **Accuracy range.** A pool where every candidate lands within a point of every other has
  no selection to study; a pool where half the candidates sit at chance suggests the
  verbalizers or the render template are broken.
- **Unparsed count.** Should be exactly zero. `allowed_token_ids` makes anything else a bug.
- **Dev/truth offset.** A few tenths of a point is normal and is subtracted in the analysis.
  Several points means the stratified item sample did not do its job.

Then run the analysis, which will not draw a figure until every check in section 9 of the PRD
passes:

```bash
!python scripts/run_analysis.py --matrices data/matrices/sst2__*.npz
```
