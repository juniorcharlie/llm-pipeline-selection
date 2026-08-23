# Novelty check — the go/pivot gate

**This is the highest-risk item in the plan and it blocks everything else.** It could not be
run from the sandbox that produced this repo: `arxiv.org`, Google Scholar, and the Semantic
Scholar API are all unreachable from there. It needs a human with a browser, and it should
happen before any GPU time is spent.

## Kill criterion

Pivot immediately if the search turns up work that already combines **all three** of:

1. measured selection bias for best-of-N prompt or pipeline selection,
2. an effective-candidate-count analysis of the correlated candidate pool,
3. budget-allocation guidance between candidate count and dev-set size.

Any two of the three leaves the paper standing. Contribution 4 (allocation) is the one that
survives being scooped on contribution 1 (measurement), so if the measurement exists but the
allocation guidance does not, reframe and lead with allocation.

## Searches to run

* \[x] "selection bias best-of-N language model evaluation"
* \[x] "prompt optimization winner's curse"
* \[x] "effective number of independent candidates prompt search"
* \[x] "how many candidates versus how many validation examples budget allocation LLM"
* \[x] "correlated candidate pool selection bias correction"
* \[x] "dev set size prompt optimizer generalization gap quantified"
* \[x] Forward-citation sweep on Cawley \& Talbot 2010
* \[x] Forward-citation sweep on the zoom correction paper (2024)
* \[x] Check whether promptolution's companion experiments already report bias vs N and m

Run 18 Aug 2026. All nine complete; see findings and verdict below.

## Record findings here

|Search|Date|Closest work found|Which of the three it covers|Verdict|
|-|-|-|-|-|
|"selection bias best-of-N language model evaluation"|18 Aug 2026|Test-time-scaling / reward-model literature on best-of-N *response* selection (e.g. "Scalable Best-of-N Selection... via Self-Certainty," arXiv:2502.18581). Different sense of "best-of-N" entirely — selecting the best sampled *response* at inference time, not reporting the winner of a pipeline/prompt search.|None of the three|Clear|
|"prompt optimization winner's curse"|18 Aug 2026|**SIREN** — Xu, Zhang, Sun, Zhou, Cao, Aggarwal (Purdue/JHU), *"Towards Reliable LLM Evaluation: Correcting the Winner's Curse in Adaptive Benchmarking,"* arXiv:2605.05973, submitted 7 May 2026. Directly names "adaptive prompt and program search" winner's curse, proposes a repeated-split, item-level Gaussian-multiplier-bootstrap correction, validates on MMLU-Pro tuning runs, and reports confidence intervals across a tuning-budget grid.|**(1) only.** It measures/corrects the bias itself, but the correction is a repeated-split bootstrap agnostic to candidate-pool structure — no spectral/moment-matching effective-candidate-count estimation (2), and the "budget grid" is cross-procedure CI comparison at a given tuning budget, not a derived optimal split between N candidates and m dev items (4). Explicitly scopes to a "fixed-shortlist, smooth stabilized selection" regime and states in its conclusion that "fully open-ended adaptive search remains future work."|**Closest match found. Does not meet the 3-of-3 kill bar, but must be cited prominently and the paper's framing adjusted (see verdict below).**|
|"effective number of independent candidates prompt search"|18 Aug 2026|Nikolopoulos, *"Spurious Predictability in Financial Machine Learning,"* arXiv:2604.15531. Near-identical statistical machinery to Estimator A: an eigenvalue/participation-ratio-style "effective multiplicity" $K\_{\\text{eff}}$ under correlated candidate search, EVT scaling validated by simulation, "Redundancy Law" section. Applied to backtested trading-strategy search, not LLM prompts.|(2) only, and in a different domain (quant finance, not LLM candidate pools)|Cite as methodological precedent for Estimator A; no domain overlap|
|"how many candidates versus how many validation examples budget allocation LLM"|18 Aug 2026|Cost-aware prompt optimizers that allocate *search* compute across candidates during optimization (MO-CAPO arXiv:2605.18869, CRAFT arXiv:2606.04661, Hyperband-based BO for prompt selection arXiv:2412.07820, PDO arXiv:2510.13907); "Instance-Optimal Estimation with Multiple LLM Judges on a Budget" (arXiv:2605.23362) allocates a budget across judges × items, not candidates × dev items. None address the *reporting-stage* question of how to split a fixed evaluation budget between number of candidates and dev-set size once search is done.|None of the three directly|Clear|
|"correlated candidate pool selection bias correction"|18 Aug 2026|Results are dominated by an unrelated meaning of "selection bias" — LLM-as-judge position/order bias in pairwise or MCQ evaluation. No overlap with statistical winner's-curse correction under candidate correlation.|None|Clear|
|"dev set size prompt optimizer generalization gap quantified"|18 Aug 2026|Confirms, does not extend, the prior-art table already in the PRD: GAAPO's train/test gap by population size (Frontiers, 2025), OPRO's §5.4 overfitting analysis (arXiv:2309.03409), "Understanding prompt engineering may not require rethinking generalization" (arXiv:2310.03957). All report isolated observations of a generalization gap; none map bias across an (N, m) grid or compare correction methods.|(1), informally and already cited|Clear, no new risk|
|Forward-citation sweep — Cawley \& Talbot 2010|18 Aug 2026|\~2,100 citations, overwhelmingly classical ML / bioinformatics (nested cross-validation, protein inference). No LLM prompt-pool application surfaced.|None|Clear|
|Forward-citation sweep — zoom correction (Zrnic \& Fithian 2024, arXiv:2411.18569)|18 Aug 2026|No LLM-prompt-pool application found. Notably, SIREN (above) does *not* cite or build on the zoom correction — it derives its own repeated-split/bootstrap machinery from a different lineage (Andrews, Kitagawa \& McCloskey "Inference on Winners," QJE 2024; Berk et al. 2013 valid post-selection inference).|None|Clear|
|Check whether promptolution's companion experiments report bias vs N and m|18 Aug 2026|promptolution (arXiv:2512.02840) is a framework/tooling paper (modular prompt-optimizer library, benchmarking use-cases). No bias-vs-(N, m) measurement study among its reported experiments.|None|Clear, confirms PRD's assumption|

### Adjacent finding worth citing, not gating

Berman et al., *"Valuing Winners: When and How to Correct for Selection Bias in Randomized Experiments,"* arXiv:2605.18887 (May 2026). Not LLM-related, but structurally close to Section 8–9 of the PRD: it evaluates a near-identical slate of corrections (bootstrap, m-out-of-n bootstrap, sample splitting, cross-fitting, shrinkage/plug-in) against the same three targets the PRD proposes (bias, MSE, 90%/95% coverage). Strengthens the "framing sentence" in Section 3 — the statistical machinery really is this well-trodden outside LLM contexts — and is a good related-work citation for the correction-method comparison in Section 8.

## Verdict — GO, with a required reframe

**Kill criterion not met.** No single work combines all three of (1) measured selection bias for best-of-N prompt/pipeline selection, (2) an effective-candidate-count analysis of the correlated pool, and (3) budget-allocation guidance between candidate count and dev-set size. SIREN (arXiv:2605.05973) covers (1) alone and explicitly leaves the correlated-pool structure and open-ended search as future work.

**But SIREN changes the risk profile of Contribution 1.** It is very recent (7 May 2026), it names "the winner's curse in adaptive \[prompt] benchmarking" directly, and it already has an MMLU-Pro empirical demonstration that naive winner-reporting is optimistic. Simply reporting "bias exists and is large" is no longer a novel headline. Two changes follow, consistent with the PRD's own contingency plan in Section 5's aside and Section 13's risk table:

1. **Cite SIREN in the first paragraph of related work**, alongside Cawley \& Talbot and the zoom correction, and state explicitly what it does not do: no correlation-structure characterization of the candidate pool, no effective-candidate-count estimator, no N-vs-m allocation guidance, and an explicit scoping to fixed-shortlist smooth selection that this paper's random-search setting does not need to assume.
2. **Lead the abstract and framing even more heavily with Contributions 2–4** (mechanism, correction via `N\_eff`, allocation guidance) rather than Contribution 1 (bare measurement) — the PRD's aside under Section 5 already recommended this as the contribution that "survives being scooped," and SIREN is exactly the kind of partial scoop that recommendation anticipated.

No changes to the schedule, task list, or kill criteria are needed. Proceed to the Wed 19 Aug matrix build.

## Fallback if the gate closes

The attribution audit: reimplement published prompt-optimizer mechanisms as isolated toggles
on one shared codebase and ablate them, claiming that of N published mechanisms only k survive
isolation. Lower risk, weaker result, more implementation work — but it reuses the same
correctness-matrix infrastructure this repo already contains, so the pivot cost is close to
zero. `CorrectnessMatrix`, the resampling harness, the statistical protocol, and the
figure pipeline all carry over unchanged.

