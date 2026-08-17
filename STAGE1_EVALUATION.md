# Stage-1 evaluation: what the DINO encoder learned, and what to change next

Report on `outputs/pretrain_swinv2_dino/dino_backbone_epoch_0100.pth` — the
SwinV2-Small trunk produced by 100 epochs of DINO self-distillation on the seed
crops, and the encoder every stage-2 run is about to load.

Everything below is produced by `python main.py eval-pretrain` and is
reproducible from the artifacts in `outputs/eval_pretrain/` (metrics, tables,
figures, cached features, checkpoint digests). How the evaluation works and why
these measurements were chosen:
[`architecture/08_STAGE1_REPRESENTATION_EVALUATION.md`](architecture/08_STAGE1_REPRESENTATION_EVALUATION.md).

> **Every number below is traceable to `outputs/eval_pretrain/metrics.json`** and,
> for the readout figures, recomputable from `out_of_fold_predictions.npz`.
> `provenance.json` records the checkpoint SHA-256s, the git commit and the library
> versions the run used.

**The three findings that matter most, up front:**

1. The 13.34-hour stage-1 run **did not improve any label readout over the
   ImageNet-1k weights it started from** (+0.31 pp on the 27-way linear probe,
   −0.69 pp on k-NN, against a ±10 pp fold SD). It did move the representation
   substantially (linear CKA 0.297) — just not in a direction the task rewards
   (Findings 2.1–2.3).
2. **91.6 % of that wall clock was the GPU waiting for the dataloader**
   (`data.num_workers=0`). The same run is available in ~1.5 h (Finding 1.3).
3. **18.65 points** of a crop-level sub-variety accuracy on this dataset is
   near-duplicate matching rather than discrimination, and **5 of 27 sub-varieties
   cannot be tested at all** under a photograph-disjoint protocol because they come
   from one photograph each (Findings 2.4, 2.7).

---

## Part 1 — The run, from its own event stream

Recovered from `outputs/hydra/2026-08-14/17-25-07/events.jsonl` (1,692 metric
records, 1,270 logged steps, 100 epochs). Figure:
`figures/fig01_pretrain_dynamics.png`.

### What was actually run

| | Value | Note |
| --- | --- | --- |
| Trunk | `swinv2_small_window16_256`, 48.96 M | ImageNet-1k `ms_in1k` init, unfrozen. **Small, not the Tiny several docs still describe** (R12) |
| DINO head | 2.63 M, 768→1024→1024→256→2048 | 0.53 M of it prototypes |
| Views | 2 global @ 256 px + 4 local @ 101 px→256 | 10 cross-view terms per sample |
| Physical batch | 64, accumulation 1 | 127 steps/epoch over 8,173 images |
| Learning rate | 1.25e-4 | derived: 5e-4 × 64/256 |
| Schedule | 10-epoch linear warmup, then cosine over 90 to 0 | one `SequentialLR` |
| Teacher | momentum 0.996→1.0, τ_t 0.04→0.07 over 30 epochs | |
| Centering | Sinkhorn-Knopp, 3 iterations, K = 2048 | KoLeo λ = 0.1 |
| Precision | bf16 autocast, `torch.compile`, 48 attention modules on SDPA | H100 PCIe 80 GB |
| Wall clock | **13.34 h** | 415.7 PFLOPs estimated |

### Finding 1.1 — The objective converged by epoch 20; the remaining 80 epochs bought 8.7 %

| Epoch | 1 | 10 | 20 | 25 | 50 | 75 | 90 | 100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Loss | 7.557 | 6.434 | 5.816 | 5.815 | 5.754 | 5.672 | **5.649** | 5.649 |

Total improvement 1.907 nats. Of that, **1.741 (91.3 %) happened in the first 20
epochs**, 0.062 in epochs 20→50, and 0.104 in epochs 50→100. The minimum is at
epoch 90 and epoch 100 is 0.0005 nats *worse*; the last-quarter improvement is
0.0185 nats.

That is a converged run, and it means **adding epochs to this exact recipe is the
one change that is certain not to help.** The relevant question is whether the
epochs that were spent produced a better *representation* even after the loss
stopped moving — which the loss cannot answer and Part 2 can.

### Finding 1.2 — No collapse, by any of the four available measures

| Measure | Value | Reference | Verdict |
| --- | --- | --- | --- |
| Teacher entropy `H` | 7.053 → 5.374 nats | floor `log(K/B_t)` = 2.773, ceiling `log K` = 7.625 | 1.94× the structural floor, 70.5 % of the nominal ceiling. Healthy |
| Soft prototype occupancy | 0.994 of K (mean), min 0.974 | 1.0 = perfectly uniform marginal | Sinkhorn is doing its job |
| KL(batch marginal ‖ uniform) | 5.7e-3 mean, 2.7e-2 max | 0 | negligible concentration |
| KoLeo term | 0.369 → 0.030 | lower = larger nearest-neighbour distances | 12× spread increase |

The entropy trajectory is the informative one: it falls fast for ~2,000 steps
(≈ epoch 16), reaches 5.22, and then sits at 5.37 for the remaining 84 epochs.
Sharpening stopped when the loss did.

**Do not read the soft occupancy as "the prototypes specialised".** It is the
teacher's *soft* marginal inside a 128-view micro-batch, which Sinkhorn forces
towards uniform by construction. The hard argmax over the dataset tells a
different and complementary story — see Finding 2.6.

### Finding 1.3 — The GPU was idle for 91.6 % of the run

`epoch/data_wait_fraction`: mean **0.916**, min 0.866, max 0.928. Throughput 17.0
images/s mean (14.2–18.8). Peak VRAM 43.5 GB of 79.2 GB.

The cause is in the launch command: `data.num_workers=0`. Stage 1 builds **six
independent PIL pipelines per sample** (random resized crop, colour jitter,
grayscale, blur, solarize, normalise); at batch 64 that is 384 augmentation chains
per step, and with zero workers all of them run in the process that is also
supposed to be feeding the GPU. `cache_images=true` removed the *decode* cost
(8,173 images cached in 91.3 MB) but not the augmentation cost.

Arithmetic on the consequence:

- GPU-busy time ≈ 13.34 h × (1 − 0.916) = **1.12 h**.
- 415.7 PFLOPs / 4,032 s ≈ 103 TFLOP/s, ≈ 14 % of an H100 PCIe's bf16 peak — a
  reasonable figure for many small window-attention kernels.
- So the run had **~12× of wall clock available for free**, and a second one at
  57 % VRAM headroom.

This is the single largest finding in the report, and it is not about the model.

### Finding 1.4 — Three settings were left where a 12× stall makes them expensive

- **Gradient norm** averaged 1.635 and exceeded the clip of 3.0 in 1.6 % of logged
  steps. Nothing to fix; the clip is not distorting training.
- **Weight decay** ramped 0.04 → 0.4 as designed, on the correct parameter group.
- **The local crops are close to information-free.** Median source crop is 52×51
  px. A local view samples 5–40 % of *that* area — 12–33 px of real content — and
  is then upsampled to 256 px. Four of the six views per sample are therefore
  mostly interpolation. `match_view_lowpass` exists in
  `src/datasets/transforms.py` precisely to remove the resulting resolution
  shortcut by giving the global views the same artefact, and it was **off**.

---

## Part 2 — What the representation actually learned

Protocol, stated once. The headline numbers are **out-of-fold**: 5-fold
`StratifiedGroupKFold` over all 9,357 crops, each fold's held-out half disjoint by
source photograph, every crop held out exactly once, all 27 classes present. The
single held-out split stage 2 uses is reported beside it for comparability and is
*not* the headline, for the reason in Finding 2.7.

Chance is 3.70 % for 27 sub-varieties and 25.0 % for 4 seed types (a
support-weighted majority-class baseline is 5.2 % / 47.4 %).

### Finding 2.1 — In-domain self-distillation did not improve the readout over the ImageNet weights it started from

`tables/encoder_comparison.csv`, `figures/fig02_readout_comparison.png`.
Out-of-fold over all 9,357 crops; `±` is the SD across the five
photograph-disjoint folds.

| Encoder | probe 27-way | probe, 22 testable | macro F1 | k-NN 27-way | probe 4-way | k-NN 4-way |
| --- | --- | --- | --- | --- | --- | --- |
| `dino_epoch100` (shipped) | 0.6284 ± 0.1016 | 0.7378 | 0.5786 | 0.5003 | 0.9839 | 0.9864 |
| `dino_epoch100_teacher` (EMA) | 0.6285 ± 0.1029 | 0.7379 | 0.5788 | 0.5028 | 0.9846 | — |
| `dino_epoch50` | **0.6358** ± 0.1006 | **0.7464** | 0.5854 | 0.5017 | 0.9851 | 0.9857 |
| `dino_epoch25` | 0.6276 ± 0.1004 | 0.7368 | 0.5781 | 0.5019 | 0.9858 | 0.9870 |
| **`imagenet_init`** (the start point) | **0.6253 ± 0.0872** | **0.7341** | 0.5723 | **0.5072** | 0.9832 | 0.9788 |
| `random_init` (floor) | 0.3804 ± 0.0587 | 0.4465 | 0.3385 | 0.3292 | 0.8441 | 0.7964 |
| chance | 0.0370 | 0.0455 | — | 0.0370 | 0.2500 | 0.2500 |

Read the `dino_epoch100` and `imagenet_init` rows together. **The 13.34-hour
stage-1 run moved the 27-way probe by +0.31 pp over the ImageNet-1k initialisation
it started from (+0.37 pp on the 22 testable classes), against a fold-to-fold SD of
±10 pp — and moved the parameter-free k-NN by −0.69 pp, i.e. the wrong way.** The
4-way seed-type readout moved +0.07 pp. None of these differences is resolvable by
this dataset, and two of the three have the wrong sign.

**The EMA teacher is not the missing piece either.** DINO's own convention is to
ship the teacher rather than the student, and this pipeline publishes the student —
so the comparison had never been made. Extracted from the resume checkpoint and
evaluated on identical footing, the teacher scores **+0.01 pp** on the probe and
**+0.25 pp** on k-NN. It is free to switch (the weights already exist) and it
changes nothing.

The measurement is not blind: `random_init` sits 24.5 pp below both, so the probe,
the split, and the pipeline all have ample dynamic range. Decomposing what the
encoder is worth on this task:

| Contribution | 27-way probe |
| --- | --- |
| architecture + linear readout alone (`random_init`) | 0.3804 |
| **+ ImageNet-1k supervised pretraining** | **+0.2449** |
| **+ 100 epochs of in-domain DINO** | **+0.0031** |

Two caveats, both stated so the finding is not overread:

- **A probe is not the whole story.** Stage 2 does not fit a linear readout; it
  fits a 9 M-parameter MoE head, and a representation that helps that head could
  in principle be invisible here. But the k-NN, the clustering (Finding 2.2) and
  the low-shot curve (Finding 2.5) all agree with the probe, which is a
  considerable amount of independent evidence pointing the same way.
- **The encoder did change.** Linear CKA against the ImageNet initialisation is
  **0.297** — a very large move in representation space, not a stalled run. Stage 1
  did something; it just did not do something the task rewards.

### Finding 2.2 — The geometry improved, the *cluster* structure got worse, and one of those matters more than the other

| Measure | `dino_epoch100` | `imagenet_init` | `random_init` | Direction |
| --- | --- | --- | --- | --- |
| RankMe | 314.6 | 319.9 | 26.3 | higher = more directions used |
| Participation ratio | **18.31** | 9.13 | 1.36 | higher = variance spread more evenly |
| Stable rank | 6.03 | 3.47 | 1.18 | higher = less dominated by one direction |
| Dimensions for 95 % variance | 110 | 144 | 3 | — |
| Top-1 variance share | 0.166 | 0.289 | 0.851 | lower = less anisotropic |
| ‖mean of unit vectors‖ | 0.541 | 0.730 | 0.793 | lower = less common direction |
| Dead channels (of 768) | 4 | 4 | 0 | — |
| Silhouette, 27-way cosine | **+0.043** | +0.030 | −0.209 | higher = better separated |
| k-means NMI, k = 27 | **0.490** | **0.569** | 0.411 | higher = better |
| k-means cluster accuracy, k = 27 | **0.333** | **0.460** | 0.267 | higher = better |
| k-means NMI, k = 4 | **0.180** | **0.677** | — | higher = better |
| Retrieval P@1 (same-photograph excluded) | **0.475** | 0.456 | 0.285 | higher = better |
| Uniformity (Wang & Isola) | −3.54 | −2.15 | — | lower = more uniform |
| Alignment (positive pairs) | 1.111 | 0.638 | — | lower = more invariant |

There is no collapse anywhere: RankMe 314.6 of a possible 768, 110 directions to
reach 95 % of the variance, the mean direction *weaker* than ImageNet's, and the
participation ratio doubled. Whatever went wrong, "the features collapsed" is not
it.

What DINO actually did is legible in this table: it **spread the representation
out**. Participation ratio 9.1 → 18.3, top-1 variance share 0.29 → 0.17, uniformity
−2.15 → −3.54. That buys a small real gain in nearest-neighbour retrieval
(P@1 0.456 → 0.475 with same-photograph neighbours excluded) and in silhouette
(+0.030 → +0.043).

And it **cost the coarse cluster structure**. k-means at k = 4 against the four seed
types falls from NMI 0.677 to 0.180 and cluster accuracy 0.668 to 0.372; the
seed-type silhouette falls 0.293 → 0.014. The 4-way *probe* is unchanged at 0.98,
so the information is still there and still linearly decodable — it is simply no
longer arranged as four clusters. For a pipeline whose stage-2 head starts with a
coarse seed-type classifier and feeds its posterior into the routing (Eq. 6, Eq. 9),
that is the opposite of the direction one would choose.

> **A methodological caution the table makes visible.** `fisher_ratio` (0.33 DINO /
> 0.96 ImageNet / **2.38 random**) and `calinski_harabasz` (74.7 / 221.4 /
> **547.1**) both *rank the untrained encoder first*. They are ratios of
> between-class to within-class scatter, and a representation living in three
> effective dimensions inflates both. They are reported for completeness and should
> not be used to compare representations of different effective rank — which is
> exactly what RankMe is for.

### Finding 2.3 — Epochs 25 → 100 changed nothing measurable either

| Stage-1 epochs | probe 27-way | k-NN 27-way | silhouette | RankMe | k-means NMI |
| --- | --- | --- | --- | --- | --- |
| 25 | 0.6276 | 0.5019 | 0.0377 | 335.6 | 0.469 |
| 50 | **0.6358** | 0.5017 | **0.0475** | 314.1 | **0.547** |
| 100 | 0.6284 | 0.5003 | 0.0433 | 314.6 | 0.490 |

`figures/fig16_milestone_progression.png`. The spread across the three milestones
(0.8 pp on the probe) is an order of magnitude below the fold SD, and epoch 50 is
nominally the best of the three. Combined with Finding 1.1 — 91.3 % of the loss
improvement inside the first 20 epochs — the conclusion is that **the last 75
epochs, about 10 hours of H100 time, produced no measurable change in the
representation's usefulness.**

This is exactly the question `experiment.training.save_epochs` exists to make
answerable, and it is answerable only because those two encoders were kept.

### Finding 2.4 — Five classes score exactly zero for a provenance reason, and the branch structure follows the provenance

`tables/per_class_sub_variety.csv`, `figures/fig12_probe_per_class_f1.png`.

The five worst classes have F1 **exactly 0.000**, and they are exactly the five
sub-varieties whose crops all come from **one** source photograph:

| Sub-variety | F1 | Support | Source photographs | Most confused with |
| --- | --- | --- | --- | --- |
| Baryard | 0.000 | 265 | 1 | PearlMillet (0.34) |
| Browntop | 0.000 | 297 | 1 | Baryard (0.71) |
| FingerMillet | 0.000 | 251 | 1 | KodoMillet (0.59) |
| PearlMillet | 0.000 | 294 | 1 | Baryard (0.90) |
| ProsaMillet | 0.000 | 280 | 1 | LittleMillet (0.97) |

This is not an encoder result. Under a photograph-disjoint protocol a class with
one photograph is never in the training half of the fold that holds it out, so it
cannot be predicted at all — identically for every encoder, including ImageNet and
random. Those five classes are **1,387 crops, 14.8 % of the dataset**.

Restricting to the 22 sub-varieties a photograph-disjoint protocol can test:

| Scope | crops | probe 27-way accuracy | macro F1 |
| --- | --- | --- | --- |
| all 27 classes | 9,357 | 0.6284 | 0.5786 |
| **22 testable classes** | 7,970 (85.2 %) | **0.7378** | **0.5999** |

Both are now reported by the evaluation
(`oof_probe_sub_accuracy_testable_classes`), because the first understates every
encoder by ~11 points for a reason unrelated to the encoder, and the second hides
that a fifth of the taxonomy is unmeasurable.

The per-branch numbers follow the same logic — the worst branch is the one with the
worst provenance:

| Seed type | sub-varieties | out-of-fold accuracy | macro F1 |
| --- | --- | --- | --- |
| Rice | 13 | 0.780 | 0.595 |
| Mustard | 3 | 0.648 | 0.210 |
| Amaranthus | 3 | 0.472 | 0.235 |
| **Millet** | 8 (5 of them single-photograph) | **0.413** | **0.120** |

The best classes are all rice (PM30 0.991 — actually mustard, then KarurKuruvai
0.971, MilaguSamba 0.967, SivapuKavuni 0.957), and the remaining genuine confusions
are within-branch and visually plausible: Jagnath ↔ Poosa33 (two mustards, 0.53 /
0.52 either way) and AMT-4 → AMT-2 (0.38). `figures/fig20_retrieval_examples.png`
shows why: the nearest neighbours of a query are near-identical grain at 52 × 51 px,
frequently from a different variety.

The two independent readouts agree on the parent seed type for **99.27 %** of crops
(per branch: Amaranthus 0.999, Mustard 0.997, Rice 0.999, Millet 0.978), and the
sub-variety macro-OvR AUC is 0.817. The hierarchy is present and consistent; the
difficulty is entirely within-branch.

`figures/fig13_class_centroid_similarity.png` shows the same thing directly, and it
is the one figure that speaks to whether stage 2's design assumption holds. With the
27 centroids ordered under their parents, all four blocks are visibly warmer inside
than outside and the rice block is clearly separated from the other three — so
**the taxonomy is genuinely present in the frozen representation**, which is what the
hierarchical KL term (Eq. 10) and the seed-conditioned router (Eq. 6 → Eq. 9) assume.
It also shows why the residual errors are what they are: within Amaranthus and within
Mustard the centroids sit above 0.95 cosine of each other, which is the same fact as
`AMT-4 → AMT-2` and `Jagnath ↔ Poosa33`.

### Finding 2.5 — In the low-label regime the ImageNet features are *better*

`figures/fig14_low_shot_curve.png`, mean ± SD over 5 independent label draws.

| Labels per class | `dino_epoch100` | `imagenet_init` |
| --- | --- | --- |
| 1 | 0.2287 ± 0.0238 | **0.2757 ± 0.0512** |
| 2 | 0.3043 ± 0.0529 | **0.3189 ± 0.0298** |
| 5 | 0.4238 ± 0.0142 | **0.4516 ± 0.0403** |
| 10 | **0.5027 ± 0.0164** | 0.4915 ± 0.0205 |
| 25 | **0.5683 ± 0.0153** | 0.5652 ± 0.0062 |

The individual gaps are within about one SD, but the *ordering is consistent* at 1,
2 and 5 shots: the ImageNet initialisation is the better few-shot representation,
and DINO only catches up once there are 10+ labels per class. The practical case
for in-domain SSL — "it pays off when labels are scarce" — is not supported here.

### Finding 2.6 — DINO's own prototype head learned the taxonomy better than its trunk did

| Measure | Value |
| --- | --- |
| Active prototypes (win ≥ 1 argmax over 9,357 crops) | **1,291 of 2,048** (63.0 %) |
| Usage entropy | 6.65 nats, 0.872 normalised; perplexity 771.6 |
| Largest single prototype share | 0.0091 |
| NMI vs sub-variety | **0.5095** |
| Purity vs sub-variety | 0.6599 |
| Bottleneck (256-D) silhouette, test split | **+0.172** |
| Trunk (768-D) silhouette, same test split | +0.115 |

The prototype layer is healthy — 63 % of prototypes in use, an effective 772 of
2,048, no prototype above 1 % of the mass — and it carries real label information
(NMI 0.51, comparable to the k-means NMI of 0.49 on the trunk features).

The more useful result is the last two rows. The DINO head's L2-normalised
**bottleneck** is better clustered by sub-variety than the trunk output stage 2
consumes (+0.172 vs +0.115 on the same crops), in 256 dimensions rather than 768,
with 45 directions carrying 95 % of its variance. Stage 1 **discards this space**:
only `student_backbone` crosses to stage 2. That is a defensible design — the head
is trained to serve the prototype task — but it is now a measured trade rather than
an assumption, and it points at R11 below.

Note the contrast with the training log, which reports `prototype_utilization ≈
0.994` throughout. That is the *soft* teacher marginal inside a 128-view
micro-batch, which Sinkhorn forces towards uniform by construction. The 63 % here is
the hard argmax over the dataset. Both are true; neither is the other.

### Finding 2.7 — The split protocol moves the headline number 60 × more than the encoder does

Same encoder, same probe, same seed — only the split protocol differs:

| Protocol | probe 27-way | k-NN 27-way | probe 4-way |
| --- | --- | --- | --- |
| grouped (photograph-disjoint) | 0.6500 | 0.5099 | 0.9980 |
| stratified (crop level, as submitted) | **0.8365** | **0.6891** | 0.9957 |
| **delta** | **+18.65 pp** | **+17.92 pp** | −0.23 pp |

`figures/fig21_split_protocol_delta.png`. **18.65 points of a crop-level
sub-variety accuracy on this dataset is near-duplicate matching**, not sub-variety
discrimination — the same physical seeds, the same lighting, overlapping bounding
boxes, on both sides of the boundary. For scale: the entire contribution of stage 1
measured above is 0.31 pp.

Retrieval shows the same effect with no readout involved at all: P@1 falls from
0.664 to 0.475 once same-photograph neighbours are excluded.

The related protocol defect is that the *grouped* single split is itself unbalanced
— it holds 14 of 27 classes (see §3 of the architecture note) — which is why the
out-of-fold protocol exists and why Finding 2.4's restricted figure is reported.

### Finding 2.8 — Calibration: over-confident, fixable, but not by the standard protocol

| Quantity | Value |
| --- | --- |
| ECE, raw | 0.1294 |
| Mean confidence vs accuracy | 0.767 vs 0.638 (over-confident by +0.129) |
| ECE after the validation-fitted temperature (T = 3.10) | 0.2744 |
| ECE at the oracle temperature (T = 1.50, chosen on test) | **0.0584** |

`figures/fig19_probe_reliability.png`. The probe is over-confident, and one
temperature *can* cut its ECE by 55 % (0.129 → 0.058 at T = 1.50) — but the
temperature fitted on a photograph-disjoint validation fold overshoots to 3.10 and
makes calibration **worse than doing nothing** (0.274). The folds differ in class
composition (Finding 2.4), so the confidence/accuracy relationship does not transfer
between them.

Reported as three numbers rather than one because the alternative — quoting only
"ECE 0.27 after temperature scaling" — would present a protocol failure as a
property of the encoder. The oracle row is labelled as unachievable in deployment;
it exists to separate *calibratable* from *calibrated*.

### Finding 2.9 — The most discriminative features are not the ones stage 2 reads

`figures/fig15_layerwise_probe.png`, `C` selected per stage on the validation fold.

| Trunk stage (width) | `dino_epoch100` | `imagenet_init` |
| --- | --- | --- |
| stage 1 (96) | 0.5807 | 0.5716 |
| stage 2 (192) | 0.5930 | 0.5706 |
| **stage 3 (384)** | **0.6857** | **0.6403** |
| stage 4 (768, pre-norm) | 0.5762 | 0.5395 |
| pooled final output (768, post-norm) — *what stage 2 consumes* | 0.6500 | 0.6337 |

Stage 3 beats the pooled final output by **+3.6 pp** for the DINO encoder and
+0.7 pp for ImageNet. The final stage's own pre-norm features are the *worst* of the
four. That is a familiar pattern — the last stage of a pretrained trunk specialises
towards its pretraining objective — and it is actionable: stage 2 reads only the
final `8×8×768` grid, and a 384-channel stage-3 grid (which is `16×16`) is both more
discriminative here and closer to the paper's `z ∈ ℝ³⁸⁴` width.

### Finding 2.10 — Augmentation invariance did not improve either

| Measure | `dino_epoch100` | `imagenet_init` |
| --- | --- | --- |
| Alignment `E‖u−v‖²` over two global views (lower = better) | 1.111 | **0.638** |
| Uniformity (lower = better) | **−3.538** | −2.149 |
| Cosine: two augmented global views of the same crop | 0.445 | 0.681 |
| Cosine: same sub-variety, different crop | 0.462 | 0.762 |
| Cosine: different sub-variety | 0.290 | 0.522 |
| **same-image minus same-class** | **−0.029** | **−0.039** |
| Clean-vs-augmented cosine (same crop) | 0.434 | 0.724 |
| **Self-retrieval @1** (augmented view retrieves its own clean view) | **0.311** | 0.297 |

`figures/fig17_augmentation_invariance.png`. The row to read is
**`same-image minus same-class`, and it is negative for both encoders.** Two
augmented views of the *same* 52 × 51 px crop are *less* similar in feature space
than two different crops of the same variety — for DINO by 0.029, for the ImageNet
initialisation by 0.039.

That is the mechanism this whole report has been circling. DINO's objective asks the
student to match a pair of views that the augmentation has pushed further apart than
two genuinely different instances. Over 100 epochs it did make progress on exactly
that quantity — the gap narrowed from −0.039 to −0.029, and self-retrieval@1 rose
0.297 → 0.311 — but the invariance it was buying is dominated by crop, blur and
colour nuisance rather than by which seed is in the picture. **The objective was
optimised; it was optimising the wrong invariance for this data.**

Note also why the raw cosines must not be compared *across* the two columns: DINO's
uniformity is −3.54 against ImageNet's −2.15, and spreading features over the sphere
lowers every cosine at once. The signed gaps within a column, and the scale-free
self-retrieval score, are the comparable quantities — which is why the figure is two
panels rather than six overlaid histograms.

### Finding 2.11 — Inference cost

Measured by the repository's own profiler on this machine (Apple M-series, MPS,
fp32, `deterministic=True`):

| | Value |
| --- | --- |
| Parameters | 48.96 M |
| GFLOPs / sample | **25.56** (measured by dispatch, matching the training run's 25.57) |
| Latency, batch 1 / 8 / 32 | 49.0 / 47.6 / 43.8 ms per sample |
| Throughput, batch 32 | 22.8 samples/s |
| Peak allocation | 228 MB |

The GFLOPs figure is the one that transfers off this machine, and it is where the
trunk choice shows: SwinV2-Small is 25.56 GFLOPs/view against Tiny's 13.32 for the
same `8×8` grid and the same 768 channels. Given Findings 2.1–2.3, **the extra
capacity is not paying for itself on this dataset** — which makes Tiny a
defensible default for the next run and halves the stage-1 compute.

---

## Part 3 — What to change, in order of expected return per unit of effort

Each item states the change, the evidence for it, the expected effect, and how to
falsify it. Nothing here is a guess dressed as a recommendation: where the
evidence does not yet exist, the item says what to measure instead of what to do.

**The framing that follows from Part 2.** Stage 1 as configured is not
underconverged, not collapsed, and not undertrained — it is optimising an objective
whose invariance target is wrong for 52 × 51 px crops (Finding 2.10), on a dataset
with 81 scenes of diversity (Finding 2.4), starting from weights that already solve
most of the task (Finding 2.1). So the useful changes are, in order: **stop paying
12× for the wall clock (R1)**, **fix what the objective is asked to be invariant to
(R3)**, and **stop spending epochs and capacity that measurably buy nothing (R2
and R11)**. Two of the recommendations in the first draft of this report were
*refuted* by the measurements and are marked as such.

### R1 — Set `data.num_workers`. Twelve times the wall clock, zero risk to the objective

**Evidence:** Finding 1.3 — `data_wait_fraction` 0.916 mean, and the launch command
carried `data.num_workers=0`.

**Change:**

```bash
python main.py pretrain data.num_workers=8   # or omit it: the config default is "auto"
```

`auto` resolves to affinity-aware cores per rank capped at 8. Keep
`cache_images=true` — under `fork` (Linux) the decoded buffer is shared
copy-on-write across workers, and it already removed the *decode* cost; the
remaining cost is the six PIL augmentation chains per sample, which is what the
workers parallelise.

**Expected:** 13.3 h → 1.5–2.5 h for the identical 100 epochs. The objective is
untouched: workers change who computes an augmentation, not what it is.

**Falsify:** `scripts/bench_pretrain_step.py --scaling 1` before and after, and
`epoch/data_wait_fraction` in the first epoch of the real run. If it does not fall
below ~0.3, the bottleneck is elsewhere and everything below should be re-costed.

Everything that follows depends on this one, because it converts "one 100-epoch
run per 13.5 h" into "five configurations per day".

### R2 — Do not add epochs to this recipe

**Evidence:** Finding 1.1. 91.3 % of the loss improvement happened in the first 20
epochs; the minimum is at epoch 90 and epoch 100 is marginally worse. Finding 2.3
says whether the *representation* kept improving after the loss stopped.

**Confirmed by Finding 2.3.** Epoch 25 / 50 / 100 score 0.6276 / 0.6358 / 0.6284 on
the probe — a spread an order of magnitude below the fold SD, with epoch 50
nominally best. The loss plateau and the representation plateau agree.

**Change:** `epochs=50` with `save_epochs=[10,25,50]`. Combined with R1 that is a
**~45-minute** stage 1 instead of 13.34 hours, at no measured cost in
representation quality. Spend the freed budget on R3's arms and on 3–5 seeds, so
component claims get error bars.

**Do not** follow Table 1's 300 epochs. Nothing in this evaluation supports it, and
Finding 1.1 shows the loss minimum is already at epoch 90 of 100.

### R3 — Fix what the objective is asked to be invariant to. **This is the one substantive change.**

**Evidence, and it is direct.** Finding 2.10: two augmented global views of the same
crop are **less** similar in feature space than two different crops of the same
variety (`same-image minus same-class` = −0.029 for DINO, −0.039 for ImageNet). The
augmentation destroys instance identity, so Eq. 1 spends its capacity re-learning
crop/blur/colour invariance instead of seed identity. Supporting evidence: the crops
have a median size of **52 × 51 px**, and `local_crops_scale: [0.05, 0.4]` samples
5–40 % of *that* area — 12 to 33 px of real content — before upsampling to 256 px, so
four of the six views per sample are mostly interpolation.

**Change** — three independent knobs, all one token, all already implemented:

```bash
# 1. keep real signal in the local views (the largest effect expected)
python main.py pretrain 'data.augmentation.local_crops_scale=[0.25,0.6]'

# 2. remove the resolution shortcut: give the global views the same low-pass
#    artefact the local ones necessarily carry
python main.py pretrain data.augmentation.match_view_lowpass=true

# 3. stop destroying colour, if colour is a real cue here (see R7)
python main.py pretrain data.augmentation.color_jitter_prob=0.3
```

**Expected:** a *higher* DINO loss (the pretext task gets harder) together with a
*less negative* `same_image_minus_same_class` and a better probe. The loss and the
readout move in opposite directions, which is exactly why this cannot be assessed
from the training log and why `eval-pretrain` exists.

**Falsify:** four 50-epoch arms (baseline + the three above), `eval-pretrain` on each,
compare `oof_probe_sub_accuracy_testable_classes` and
`invariance/same_image_minus_same_class`. With R1 that is ~45 min per arm — three
hours for the whole design, against the 13.34 hours the current single arm cost.

### R4 — Raise the physical batch into the idle 45 % of VRAM

**Evidence:** peak 43.5 GB of 79.2 GB at batch 64.

**Change:**

```bash
python scripts/bench_pretrain_step.py --find-batch-size 64,96,112,128
python main.py pretrain data.batch_size=112 experiment.training.effective_batch_size=112
```

**Expected:** the benefit is *statistical*, not throughput. Sinkhorn's assignment
and KoLeo's distances are computed per micro-batch, so `B_teacher` goes 128 → 224
and the structural entropy floor `log(K/B_teacher)` falls 2.77 → 2.21 nats, opening
about 8 % more of the nominal range to measurement. Raise both keys together and
let `resolve_learning_rate` follow; do not raise accumulation instead.

### R5 — `out_dim = 2048` is roughly right; **do not shrink it on the utilisation argument**

**Refuted by measurement.** The first draft of this report expected the prototype
layer to be mostly dead and recommended `out_dim=1024`. Finding 2.6 says otherwise:
**1,291 of 2,048 prototypes (63 %) win at least one argmax**, the usage entropy is
0.872 of its maximum, effective perplexity is 772, and no prototype holds more than
0.91 % of the mass. The head is being used.

**What remains worth doing:** `K` is still coupled to the physical batch through
Sinkhorn's per-prototype evidence, so if R4 raises the batch, leave `K` alone and let
`B_teacher / K` improve on its own. Revisit `K` only if a future run's
`active_fraction` falls well below ~0.3.

### R6 — Read stage 3, not only the final stage

**Evidence:** Finding 2.9. A probe on the trunk's **stage-3** output scores 0.6857
against 0.6500 on the pooled final output that stage 2 actually consumes — **+3.6 pp,
an order of magnitude more than stage 1's entire contribution** — and the same
ordering holds for the ImageNet encoder (0.6403 vs 0.6337). The final stage's
pre-norm features are the worst of the four.

**Change:** stage 2 reads the final `8×8×768` grid through
`BackboneFeatureExtractor.forward(return_tokens=True)`. Stage 3 emits a `16×16×384`
grid — already the paper's `z ∈ ℝ³⁸⁴` width, so the Eq. 4 projection would collapse
to an identity, at the cost of 256 routing tokens instead of 64. Two ways to test it
without touching the head: probe the concatenation of stage-3 and stage-4 pooled
features first (a two-line change in this evaluation), and only then decide whether
to plumb a stage selection into `BackboneFeatureExtractor`.

**Caveat:** a linear probe on a `16×16` grid's mean is not the same measurement as
grid routing over 256 tokens, so this is evidence for *trying* it, not for assuming
the gain transfers.

### R7 — Test whether the colour jitter is destroying a real cue

**Evidence:** `color_jitter_prob: 0.8` with brightness/contrast ±0.4 and
saturation ±0.2, on a task where several sub-varieties differ mainly in hue.
`conf/data/hierarchical_seeds.yaml` already flags this ("drop these back if colour
turns out to be the dominant cue"), and Finding 2.4 says which classes are being
confused.

**Change:** `data.augmentation.color_jitter_prob=0.3`, or drop
`color_jitter_saturation`/`hue` specifically.

**Falsify:** compare per-class F1 in `tables/per_class_sub_variety.csv` for the
classes that currently confuse, not the overall accuracy — a hue cue matters for a
handful of classes and will be invisible in the mean.

### R8 — The EMA teacher is **not** the missing piece. Keep shipping the student

**Refuted by measurement.** DINO's own convention is to ship the teacher rather than
the student, this pipeline publishes the student, and nothing in the repository had
ever compared them — so this looked like a free win. It is not: the teacher scores
+0.01 pp on the probe and +0.25 pp on k-NN (Finding 2.1), and marginally *worse* on
k-means NMI (0.470 vs 0.490) and silhouette (0.0426 vs 0.0433).

At `momentum_teacher: 0.996 → 1.0` over 12,700 steps the teacher's averaging window
is long enough that, on a converged run, the two networks have essentially met. The
comparison was worth making and the answer is "no change".

**What to keep from it:** `experiment.training.save_teacher_in_checkpoints=true` costs
~200 MB and makes this check a one-liner for future runs instead of requiring a
resume checkpoint to still exist. The extracted trunk is at
`outputs/pretrain_swinv2_dino/dino_teacher_backbone_epoch_0100.pth` if anyone wants
to re-test it.

### R9 — The stage-2 split protocol needs the out-of-fold option

**Evidence:** Finding 2.7 and §3 of
[`architecture/08_STAGE1_REPRESENTATION_EVALUATION.md`](architecture/08_STAGE1_REPRESENTATION_EVALUATION.md).
The grouped `GroupShuffleSplit` test set contains **14 of 27** sub-varieties, so a
stage-2 macro-F1 on it is capped near 14/27 for reasons that have nothing to do
with the model, and 13 classes contribute no test evidence at all.

**Change:** add a `grouped_cv` protocol to
`src/trainers/moe_finetune.py::split_dataset` that uses
`StratifiedGroupKFold` over the whole dataset and reports out-of-fold metrics — the
same thing `grouped_cv_readout` does here. **Not done in this change**, because it
moves every published stage-2 number and that is a decision about the paper, not a
refactor.

**Interim:** report the number of classes present in the test split alongside every
stage-2 macro-F1. The evaluation already logs it.

### R10 — The dataset is the binding constraint, and no recipe change fixes it

**Evidence:** 9,357 crops from **81 photographs** (mean 115.5 crops each); five
sub-varieties have crops from exactly one photograph; the grouped test split holds
14 classes.

**Change, outside training:** photograph each variety in **at least three separate
sessions** (different lighting, background, camera pose). That is what converts the
grouped protocol from "13 classes untestable" into a real held-out set, and it is
worth more than any hyperparameter in this document. Until then, every accuracy
figure in the paper needs the provenance caveat stated next to it, which
`summary.json` now carries automatically.

### R11 — Consider a smaller trunk, and consider keeping the DINO head's bottleneck

Two capacity observations that follow from Findings 2.1 and 2.6.

**The trunk is oversized for what it delivers.** SwinV2-Small costs **25.56
GFLOPs/view** against Tiny's 13.32 for the same `8×8` grid and the same 768 channels
(both measured). Stage 1's contribution over ImageNet is +0.31 pp; nothing in this
evaluation suggests the extra 21 M parameters are being used. `model.backbone.name=swinv2_tiny_window16_256`
(with `feature_dim: 768`, unchanged) halves stage-1 compute and is a one-token
override — and with R1 and R2 it makes the whole stage a ~20-minute job, which is the
regime in which R3's four arms become routine.

**The head's bottleneck is better clustered than the trunk output stage 2 gets.**
Finding 2.6: silhouette +0.172 in the 256-D DINO bottleneck against +0.115 in the
768-D trunk output, on the same crops. Stage 1 discards it. Worth testing:
initialise `DinoV2SwinV2Encoder`'s Eq. 4 projection from the head's first MLP layer
rather than from scratch, or probe `bottleneck → 27` directly to see how much of the
gap survives. Note the honest caveat — the bottleneck is trained to serve the
prototype task and is L2-normalised, so a better silhouette there does not
automatically mean a better input for a 9 M-parameter MoE head.

### R12 — Documentation drift to close

`conf/model/backbone/swinv2.yaml` selects SwinV2-**Small** and the published
checkpoint is a Small. `README.md`, `CLAUDE.md`, `ARCHITECTURE.md`,
`architecture/00_OVERVIEW.md` and `tests/conftest.py` were corrected as part of
this work — `SHIPPED_BACKBONE`, measured at 48.96 M parameters and 25.56
GFLOPs/view, is now pinned by
`tests/test_stage1_recipe.py::test_shipped_small_trunk_*`.

Two documents still describe the Tiny trunk in the present tense and were **left
alone deliberately**, because their figures are *derived* and need recomputing
rather than editing:

- `architecture/02_BACKBONE_AND_SSL.md` — the backbone comparison table and the
  worked budget report (27.58 M, 13.32 GFLOPs/view).
- `architecture/07_EFFICIENCY_AND_EVALUATION.md` — the 37.7 M full-model and 13.69
  GFLOPs-per-sample figures, which are Tiny + the Eq. 4 projection + the head.

Both were correct before the trunk changed. Recomputing them is a
`profile_model` run against the current config, not a search-and-replace.

---

## Reproducing this report

```bash
export SEED_DATA_ROOT=/path/to/Hierarchical_SeedData/Cropped_Samples
export SEED_OUTPUT_DIR=/path/to/outputs
python main.py eval-pretrain
```

Determinism: `seed: 42` everywhere, TF32 off, `deterministic=True`,
`matmul_precision="highest"`, no autocast, every sklearn call seeded, features
extracted in fp32. `provenance.json` records the checkpoint SHA-256s, the split
manifest digest, the git commit, the resolved config and the library versions.
`features/*.npz` lets every figure be redrawn without a forward pass.
