# `tests/` — pytest suite

```bash
python -m pytest tests/ -q             # all 518, ~95s
python -m pytest tests/ -k arcface     # one topic
python -m pytest tests/test_models.py  # one file
```

No network access, no weight downloads, no dependency on the real dataset.

| File | Covers |
| --- | --- |
| `conftest.py` | Paper constants, synthetic dataset tree, stub encoder, shared fixtures |
| `test_moe_layer.py` | Top-2 routing, dispatch weights, sparse/dense equivalence, dormant-parameter arithmetic, the dense bypass |
| `test_components.py` | Cross-attention, ArcFace margin, projections, classifier heads |
| `test_losses.py` | Bounds, KL direction, ArcFace stability, DINO schedule, ablation toggles |
| `test_models.py` | Full cascade shapes, Eq. 8/9/11 dataflow, Top-2 routing, the four component toggles, baselines |
| `test_metrics.py` | Alignment rate, per-class metrics, AUC with absent classes, t-SNE |
| `test_representation.py` | Stage-1 metrics against cases with known answers: RankMe on collapsed vs full-rank features, probe/k-NN at chance on noise, purity vs Hungarian accuracy on over-clustering, retrieval with and without group exclusion |
| `test_pretrain_eval.py` | The evaluation stage's protocol and provenance: feature-cache digest guard, out-of-fold coverage and group disjointness, encoder-spec validation, event-stream axis separation |
| `test_efficiency.py` | Total vs. active parameters, Top-2 vs. Top-4, FLOPs, latency |
| `test_evaluation.py` | Prediction dumps, `summary.json`, the comparison CSV, publication figures |
| `test_runner.py` | Suite definitions, command construction, shared-checkpoint handling |
| `test_datasets.py` | Label hierarchy, global sub-variety indices, multi-crop pipeline |
| `test_configs.py` | Hydra composition and agreement with Table 1 / Section 5 |
| `test_integration.py` | Splits, epoch loop, checkpoint round-trip, tracker artifacts |
| `test_throughput.py` | Value-preserving speedups: SDPA parity (fp64/bf16/fp16), fused views, EMA, collate order |
| `test_distributed.py` | DDP gradient equality, distributed Sinkhorn, shared EMA centre, wrapper prefixes |
| `test_checkpointing.py` | Exact resume, atomic writes, RNG round-trip, latest-valid discovery |
| `test_precision.py` | AMP dtype selection, fp32 pinning, capability probing, reproducibility |

## Paper constants are the fixtures

`conftest.py` defines the paper's numbers once — `PAPER_EMBED_DIM = 384`,
`PAPER_NUM_EXPERTS = 6`, `PAPER_TEACHER_MOMENTUM = 0.996`, and so on. Tests
assert against these rather than against literals, so **a failure means the
implementation has diverged from the paper**, not that a test is stale.
`test_configs.py` cites the paper section in each assertion.

Routing width has **two** named constants — `SUBMITTED_TOP_K = 4` and
`REVISED_TOP_K = 2` — so a test can state which one it means.
`test_efficiency.py` compares them directly; a test asserting a bare `4` would
silently become a claim about the wrong paper.

## What the tricky tests actually check

**Top-2 routing.** That `top_k_indices` are exactly the argsort top-2 of the
gate; that the two selected weights renormalise to sum to 1; that unselected
experts get zero dispatch weight; that the full 6-wide gate still sums to 1 (the
discarded mass is what the L1 term penalises); and that sparse dispatch produces
output identical to dense evaluation.

**Routing and gradients.** With sparse dispatch, an expert that no sample routed
to receives **no gradient that step**. That is the defining property of a sparse
MoE, not a defect — but it is far more visible at `K = 2` than at `K = 4`, since
a batch of 12 fills only 24 routing slots across six experts. The tests assert
the precise invariant, `routed ⇔ has gradient`, rather than the false claim that
every expert always learns. Everything *outside* the experts is still required to
receive gradient, in every ablation.

**Dataflow, not just shapes.** `test_models.py` re-derives each intermediate and
compares: `refined_features == moe_features + projected_seed` (Eq. 9), the MoE
gate matches `moe(embedding)` directly (Eq. 8 routes on `z`), and the projection
fed logits instead of probabilities gives a *different* answer. These are the
three places the original implementation diverged from the paper.

**Component toggles.** Each of the four architectural flags is checked for four
things: the output contract still holds, gradients still flow, the model does not
*grow*, and the specific structural claim is true — `seed_projection is None`,
`attended_features == refined_features`, `gate_probs` is one-wide, and so on.
Two consequences are asserted numerically: `wo_moe` drives both MoE regularisers
to exactly 0, and `wo_residual` drives the cosine term to exactly 0 (cos(h,h) = 1).

**Active-parameter arithmetic.** Because all experts share one architecture, the
saving is a closed form and can be asserted exactly:
`dormant == (E − K) × parameters_per_expert`, and Top-2 leaves precisely twice as
many parameters dormant as Top-4 while the *total* is identical.

**Loss bounds.** Uniform expert utilisation hits exactly `-1.0`; total collapse
hits `0.0`; L1 sparsity is `0.0` when all mass is inside Top-K and equals the
discarded mass otherwise; cosine is `0.0` for identical and `2.0` for opposed
vectors. Randomised runs assert the bounds hold generally. Top-2 is shown to
discard at least as much gate mass as Top-4, and the two MoE regularisers are
shown to pull in opposite directions — the designed tension.

**KL direction.** Eq. 10 is `D_KL(P_seed ‖ P_sub)`, and the test recomputes it
by hand rather than trusting `F.kl_div`'s argument order. `use_kl_loss=False`
must lower the total by exactly the KL term, and must override a non-zero
`lambda_kl` rather than being silently re-enabled by it.

**ArcFace stability.** Logits stay within `±s` (they are `s·cos θ`, so an
unbounded value means a bug), the margin never *decreases* the loss, and
gradients stay finite at `cos θ = 1` — precisely where an `acos`-based
implementation diverges.

**DINO.** That the schedule ramps 0.02 → 0.04 over exactly 5 epochs; that the
cross-view pairing yields 10 terms for 2 teacher and 6 student views; that
centering follows `C_t = 0.9·C_{t−1} + 0.1·q̄`; and that a student matching the
teacher attains the teacher's entropy, the loss's true lower bound.

**Suite integrity.** Each ablation variant must change exactly one override; no
two variants may share an output directory; and every self-supervised-path variant must
reference the same encoder checkpoint. A suite that violated any of these would
still run to completion and produce a table that looked entirely normal.

**Reporting.** The CSV's first columns are exactly the requested ones, in order;
missing measurements are written as blanks rather than `nan`; the 27-class
confusion matrix carries all 27 unabbreviated labels on both axes; and the t-SNE
figure actually draws its cluster-name overlays.

**Overfitting one batch.** `test_integration.py` runs 30 steps on a fixed batch
and asserts the loss falls. A model that cannot overfit a single batch has a
broken gradient path, which shape assertions alone will not catch.

## Fixtures

`synthetic_dataset_root` (session-scoped) builds a real image tree matching the
paper's hierarchy — 4 seed types with 13/8/3/3 sub-varieties — at 12 images per
sub-variety. That count is chosen so a stratified 27-class train/val/test split
still leaves at least two members of every class in every partition.

`DummyEncoder` stands in for `DinoV2SwinV2Encoder`, mapping images to
`[batch, 384]` — the same contract the real encoder guarantees. Integration tests
exercise the trainer's real epoch loop without ever constructing SwinV2.

## The multi-process tests run real processes

`test_distributed.py` spawns real ranks over a real **Gloo** process group on
CPU, which is what makes it both meaningful and runnable anywhere — no GPU, no
NCCL, no launcher. The arithmetic under test is backend-independent: an
all-reduce is an all-reduce.

Each of its four claims has a failure mode that produces no error at all:

1. **DDP's averaged gradient equals the single-process gradient** on the batch
   the ranks jointly hold, including across `no_sync` accumulation. Get the
   divisor wrong and you train a different model at a loss curve that looks
   right.
2. **Distributed Sinkhorn equals single-process Sinkhorn** on the concatenated
   batch. The implementation everyone reaches for first — reduce the local
   `logsumexp` results — is off by a per-rank shift and yields an assignment that
   is *nearly* doubly stochastic, which nothing downstream notices. The test
   carries its own control: the naive per-rank concatenation must differ
   materially from the reference, so the agreement is not an accident of small
   numbers.
3. **Per-rank Sinkhorn is untouched by the presence of a process group**, which
   is what makes the default setting reproduce the single-GPU numbers.
4. **No wrapper prefix reaches a state dict.**

`test_checkpointing.py`'s headline test trains `n` steps in one process and
compares, parameter by parameter, against training `k`, saving, restoring into
**freshly constructed** objects, and training `n - k`. Fresh objects matter:
reusing them would let state survive through Python rather than through the file.
It also carries a control — `test_a_weights_only_checkpoint_is_detectably_not_enough`
— so a suite that could not distinguish a complete checkpoint from an incomplete
one would fail rather than pass quietly.

The real backend is checked separately, on the machine that will use it:
`python scripts/verify_runtime.py --gpus 2` re-runs the gradient-equality claim
over NCCL, prints the per-module SDPA parity report, and reports what the
hardware actually supports.

## Beyond the unit suite

`python scripts/dry_run.py` builds a **real** SwinV2 encoder and pushes synthetic
tensors through the entire pipeline — encoder, MoE, losses, backward pass,
metrics, efficiency profiling, CSV export. The unit tests deliberately stub the
backbone, so this is what catches a problem that only appears with the genuine
article. It needs no dataset and no checkpoint.

## Adding a test

Put paper-derived numbers in `conftest.py` rather than inline. If a test asserts
something the paper states, quote the sentence in the docstring — that is what
makes a future failure diagnosable as drift rather than as flakiness. If it
asserts something the *revision* changed, say so and name the constant.

## `test_stage1_changes.py`

Every test in this module pins a behaviour the stage-1 audit
(`STAGE1_CHANGES.md`) found either **wrong** or **unmeasurable**, and the
distinction matters when one fails:

- A **KoLeo** or **loss-decomposition** failure is a regression of a *correctness*
  fix. The pre-audit behaviour is still reachable behind a flag, so a failure here
  means the default silently moved back to it.
- A **provenance**, **split** or **nuisance** failure is a regression of an
  *instrument*. Nothing about the model changes, but a number becomes
  uninterpretable — which is how the shipped encoder came to be trained on 8,173
  of 9,357 crops with no record anywhere.

The file is organised by the audit's own section letters, so a failure names the
item it belongs to:

| Section | What it pins |
| --- | --- |
| A1 | KoLeo is applied **per global view**; the across-views form repels the two views of one image, measured on both the value *and* the gradient |
| A4 | `CE = H(q) + KL(q‖p)` holds exactly under both centerings, and the decomposition is emitted even on a non-logging step (the epoch mean has to see every micro-batch) |
| A2 / E2 | The corpus fingerprint counts what the run will read, is path-independent, and changes when one crop is removed |
| A6 | Stage 1's `summary.json` round-trips and carries the corpus |
| A7 | *(in `conftest.py`)* the fixture taxonomy matches the real tree — `Amaranthus, Millet, Mustard, Rice`, not `Millet, Mustard, Rice, Seasame` |
| B1 | `feature_stage` reads `layers.2`, reports its own width, and `forward_intermediates` matches the hook fallback bitwise |
| B3 / G1 | The `auto` worker cap is 16 and configurable |
| B4 / E6 | Raw-photograph coverage names the uncropped photographs |
| C4 | The auxiliary head is not allocated unless asked for, and never renames the published state dict |
| E1 | The handcrafted floor is ten numbers encoding size and colour |
| E5 | `grouped_cv` holds out every crop exactly once, keeps folds photograph-disjoint, and `merge_out_of_fold` restores dataset order |
| E9 | Nuisance decodability is at chance for a class-only representation and ~1 for a photograph-encoding one |
| F1 | Provenance-derived positives keep the view count and draw partners from the same photograph, deterministically in the seed |

Plus four cross-cutting groups: multi-view ordering (the collate's view-major
layout is what makes per-view KoLeo meaningful — the two facts are tied together
in one test), backward-compatible checkpoint loading, distributed execution (the
DDP wrapper must not rename the published keys), and evaluation reproducibility.

**The `conftest.py` taxonomy fix is load-bearing.** `SUBVARIETY_COUNTS` previously
used `Seasame` in place of `Amaranthus`, and because the hierarchy is built from
*sorted* directory names that put the 3-member class last instead of first: the
synthetic parent→child map was `[0]*8 + [1]*3 + [2]*13 + [3]*3` where
production's is `[0]*3 + [1]*8 + [2]*3 + [3]*13`. Every hierarchy fixture
exercised a 4-parent tree with the right totals and the wrong grouping — exactly
the class of bug the Eq. 10 KL aggregation is most likely to have. Tests that
depended on the old ordering now derive their expectations from
`SUBVARIETY_COUNTS` rather than restating them as literals, so the two cannot
agree with each other and disagree with production again.
