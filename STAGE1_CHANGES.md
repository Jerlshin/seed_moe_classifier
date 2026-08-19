# STAGE1_CHANGES.md — a codebase-specific change plan for stage-1 pretraining

**Scope.** How to make stage 1 produce a better representation *for the
photograph-disjoint 27-way seed-variety task*. No production code is changed by
this document. Every item names the exact file, the current behaviour, the
proposed change, the evidence, the expected effect, the failure mode, a
confidence level, whether it must be isolated, and how it is measured.

**Success is not a lower DINO loss.** Success is
`oof_probe_sub_accuracy_testable_classes` and `oof_probe_sub_f1_macro` under the
5-fold `StratifiedGroupKFold` protocol in `src/trainers/pretrain_eval.py`, plus
the stage-2 downstream number under `split_protocol: grouped`. Section A4 shows
that the reported DINO loss is *mostly not a learning curve at all*.

**Evidence classes used below.**

| Tag | Meaning |
| --- | --- |
| `[run]` | from `outputs/hydra/2026-08-14/17-25-07/events.jsonl` and `training.log` |
| `[eval]` | from `outputs/eval_pretrain/metrics.json` and its tables |
| `[audit]` | **measured during this audit**, from the cached features, the raw dataset, and timm; reproducible with Appendix 1 |
| `[code]` | read directly from the source, file and line cited |

---

## The seven conclusions, up front

All deltas are 27-way out-of-fold accuracy under the photograph-disjoint
protocol, measured on the *pooled* readout stage 2 currently consumes unless
stated. For scale: the entire 13.34-hour stage-1 run is worth **+0.58 pp**
`[audit]` / +0.31 pp `[eval]` over the ImageNet weights it started from.

| | Finding | Worth | Where |
| --- | --- | --- | --- |
| 1 | **The encoder was pretrained on a different corpus (8,173 crops) from the one everything downstream uses (9,357), and nothing records or checks it.** | correctness | §0.1, A2 |
| 2 | **KoLeo is applied across both global views, so it actively repels the two views of the same image** — the exact pair Eq. 1 exists to pull together. This deviates from the DINOv2 reference and is the leading candidate mechanism for `alignment` getting *worse* (0.638 → 1.111). | unknown, plausibly large | Part 1 #1, A1 |
| 3 | **Stage 2 reads the one part of the trunk that DINO consumed.** `layers.3` has CKA 0.103 against its ImageNet initialisation; `layers.2` has 0.390 and is the better readout for *every* encoder tested. | **+3.25 pp** linear, **+3.90 pp** MLP | §0.3, B1 |
| 4 | **The initialisation was never treated as a variable, and it is worth more than the whole stage-1 run.** A frozen IN-22k SwinV2-Base, with zero in-domain training, beats the shipped DINO encoder. | **+2.24 pp** (+2.50 pp at matched width) | App. 2, B2 |
| 5 | **The reported DINO loss is 95 % target entropy.** 80 % of its "improvement" is the teacher sharpening, not the student learning; the learnable part, `KL(q‖p)`, is never logged and was still falling at epoch 93. | diagnostic | §0.2, A4 |
| 6 | **Stage 1's one clearly beneficial effect is invisible to the current evaluation**: it cut within-class photograph-identity decodability from +10.0 pp above chance to +3.5 pp. | real, unreported | §0.7, E9 |
| 7 | **Ten numpy scalars score 0.5360** under the identical protocol — 15.6 pp above an untrained 49 M-parameter trunk, 9.2 pp below the shipped one. This belongs in the paper as a floor. | reporting obligation | §0.4, E1 |

**Two hypotheses this audit refuted**, and would otherwise have recommended:
absolute crop scale is *not* destroyed by the resize (it is 95–96 % linearly
decodable from both encoders; adding it explicitly moves the probe +0.10 pp), and
the colour jitter did *not* destroy the colour cue (mean RGB is **more**
decodable after DINO, R² 0.864 → 0.908). See D4 and D5.

---

> **One calibration note, and it matters for reading every table below.** The
> `[audit]` probe is my own re-implementation of the evaluation's protocol
> (`StratifiedGroupKFold(5, shuffle=True, random_state=42)`,
> `StandardScaler → LogisticRegression(C=100)`) and it lands **1.7–2.0 pp below**
> `pretrain_eval.py`'s numbers in absolute terms — it does not replicate that
> module's per-fold regularisation selection and refit. `dino_epoch100` pooled:
> `[eval]` 0.6284, `[audit]` 0.6111. **The orderings and the deltas reproduce
> exactly**, which is what every claim here rests on. Never compare an `[audit]`
> number against an `[eval]` number directly; compare within a table.

---

## Part 0 — What this audit measured that the existing evaluation did not

Seven new measurements. They are the basis for most of what follows, and two of
them *refute* recommendations that would otherwise look obvious.

### 0.1 The stage-1 run trained on a different corpus from the one everything else uses

`training.log:9` — `Cached 8173 decoded images`; `training.log:10` — `Loaded 127
batches of 64 images`. The evaluation, stage 2, and `group_report()` all operate
on **9,357** crops (`metrics.json → dataset.num_samples`). The run's config
snapshot points at `/root/Girija/Seed_Project/hierarchical_data/Cropped_Samples`
(server); the evaluation's provenance points at
`.../Dataset/Hierarchical_SeedData/Cropped_Samples` (local, verified 9,357 files
by direct count).

**The shipped encoder was self-distilled on 8,173 crops — 87.3 % of the dataset —
and no artifact anywhere records which 1,184 were missing.** `[run]` `[audit]`

### 0.2 80 % of the DINO "loss improvement" is the teacher's entropy, not learning

`loss` is a cross-entropy, so `loss = H(teacher) + KL(teacher‖student)`. Both
terms are already logged (`train/loss`, `train/teacher_entropy`).

Values are the first and last *logged step* (the evaluation's Finding 1.1 quotes
epoch means, 7.557 → 5.649; the decomposition is unchanged either way).

| | step 0 | step 12,690 | change |
| --- | --- | --- | --- |
| reported `loss` | 7.765 | 5.671 | **−2.094** |
| `H(teacher)` | 7.053 | 5.379 | −1.674 (**80.0 %** of it) |
| `KL(teacher‖student)` — the only learnable part | 0.712 | 0.293 | −0.419 (20.0 %) |

The final reported loss of 5.65 is **94.8 % irreducible target entropy**. And the
KL was *still falling at epoch 93* (minimum 0.218 at step 11,870), which the raw
loss curve — flat from epoch 20 — does not show. `[run]` `[audit]`

This does **not** overturn the "more epochs will not help" conclusion (§D1): the
epoch-25/50/100 probes are flat regardless. It makes the conclusion *stronger* —
the student kept getting measurably better at the pretext task for 80 more epochs
and the representation did not improve, which is evidence about the **objective**,
not about the budget.

### 0.3 The DINO run rewrote exactly the stage that stage 2 reads

Linear CKA between `dino_epoch100` and `imagenet_init`, per trunk stage, on the
identical 9,357 crops (`outputs/eval_pretrain/features/*.npz`):

| trunk stage | shape | CKA vs ImageNet init | 27-way OOF probe: ImageNet → DINO |
| --- | --- | --- | --- |
| `layers.0` | 64×64×96 | **0.976** | 0.5955 → 0.5976 (+0.21 pp) |
| `layers.1` | 32×32×192 | **0.960** | 0.5502 → 0.5601 (+0.99 pp) |
| `layers.2` | **16×16×384** | 0.390 | **0.6174 → 0.6436 (+2.62 pp)** |
| `layers.3` | 8×8×768 (pre-norm) | **0.103** | 0.6072 → 0.6258 (+1.86 pp) |
| pooled, post-`norm` — **what stage 2 consumes** | 768 | 0.295 | 0.6053 → **0.6111 (+0.58 pp)** |

Stages 1–2 are essentially untouched; `layers.3` is almost completely rewritten
(CKA 0.10) to serve the 2,048-prototype head. **Stage 1's gain over ImageNet is
4.5× larger at `layers.2` than at the output stage 2 reads.** `[audit]`

A nonlinear readout does not rescue the pooled features — the gap widens:

| head | ImageNet pooled | DINO pooled | ImageNet stage3 | **DINO stage3** |
| --- | --- | --- | --- | --- |
| linear probe, OOF | 0.6053 | 0.6111 | 0.6174 | **0.6436** |
| 512-unit MLP, OOF | 0.6138 | 0.6194 | 0.6432 | **0.6584** |

That directly answers *"is useful information discarded between the backbone and
stage 2?"* — **yes, about 3.3–3.9 pp of 27-way accuracy, which is 6–13× everything
stage 1 currently contributes.** `[audit]`

### 0.4 A 10-number hand-crafted baseline scores 0.536 under the same protocol

Features: `log(crop area)`, `log(aspect ratio)`, mean RGB, std RGB, mean and std
grey. Same 5-fold `StratifiedGroupKFold`, same seed, same probe.

| representation | dim | 27-way OOF acc | macro-F1 | 4-way OOF acc |
| --- | --- | --- | --- | --- |
| hand-crafted | **10** | **0.5360** | 0.4899 | 0.9468 |
| `random_init` SwinV2-Small `[eval]` | 768 | 0.3804 | 0.3385 | 0.8441 |
| `imagenet_init` `[eval]` | 768 | 0.6253 | 0.5723 | 0.9832 |
| `dino_epoch100` `[eval]` | 768 | 0.6284 | 0.5786 | 0.9839 |

An untrained 48.96 M-parameter trunk is **15.6 pp worse** than ten numpy scalars.
100 epochs of in-domain DINO on an H100 buy **+9.2 pp over the ten scalars** and
**+0.3 pp over the ImageNet weights**. This belongs in the paper as a floor row;
see §E1. `[audit]`

### 0.5 Four of every six views carry ~600 real pixels

Measured over all 9,357 crops (median 52×51 px, 51 % have both sides < 64 px):

| view | native pixels behind it (p5 / median / p95) | upsample factor to 256 px (median / p95) |
| --- | --- | --- |
| global, `scale=(0.40, 1.00)` | 675 / **2,035** / 7,101 | 5.7× / 9.9× |
| local, `scale=(0.05, 0.40)`, via 101 px | 130 / **598** / 2,608 | **10.5× / 22.1×** |
| local, proposed `(0.30, 0.70)` via 160 px | 503 / 1,482 / 4,994 | 6.7× / 11.5× |

A local view is a median 24×24 px fragment of a single seed inflated to 65,536
output pixels — 0.9 % real content. **Eight of the ten cross-view terms in Eq. 1
are anchored on such a view.** `[audit]`

### 0.6 Eighteen source photographs exist and were never cropped

`RAW_Samples/` holds **99** photographs; `Cropped_Samples/` references **81**.

| sub-variety | photos in `RAW_Samples` | photos used | unused |
| --- | --- | --- | --- |
| Poosa33 | 8 | 3 | IMG_0664–0668 |
| KodoMillet | 5 | 2 | IMG_0494, 0496, 0497 |
| LittleMillet | 5 | 2 | IMG_0504, 0505, 0508 |
| Jagnath | 5 | 3 | IMG_0671, 0673 |
| Chithrakar | 6 | 5 | IMG_0160 |
| PM30, Kullakar, SwarnaMasoori, AMT-1 | 4, 4, 4, 3 | 3, 3, 3, 2 | one each |

Cropping them raises the group count 81 → 99 (+22 %) and doubles or triples the
photograph count for four classes that currently have 2–3. It does **not** fix the
five single-photograph classes (Baryard, Browntop, FingerMillet, PearlMillet,
ProsaMillet) — each genuinely has one raw photograph. `[audit]`

### 0.7 Stage 1 *did* do one clearly useful thing: it removed most of the photograph nuisance

Nearest-class-mean decodability of the **source photograph**, restricted to
photographs *of the same sub-variety* — i.e. exactly the nuisance direction the
photograph-disjoint protocol punishes, with class identity held out of it. 22
classes have ≥2 photographs; 3-fold stratified, cosine centroids.

| encoder | within-class photograph identity | chance | above chance |
| --- | --- | --- | --- |
| `imagenet_init` | 0.4167 | 0.3167 | **+10.0 pp** |
| `dino_epoch100` | 0.3520 | 0.3167 | **+3.5 pp** |

**In-domain self-distillation cut the photograph-specific signal in the
representation by 65 %.** That is a real, correctly-signed effect that no metric
in `outputs/eval_pretrain/` reports, and it is the strongest evidence in this
audit that stage 1 is doing *something* the task should reward — it simply has not
converted into sub-variety accuracy. It also gives F1 (§F) its falsifier and the
evaluation a missing instrument (§E9). `[audit]`

---

## Part 1 — Verdict on every observed phenomenon

| # | Phenomenon | Verdict | Where |
| --- | --- | --- | --- |
| 1 | KoLeo is applied across both global views, so it repels the two views of the *same image* | **1. Confirmed problem** (deviates from the DINOv2 reference, which chunks by view for exactly this reason) | A1 |
| 2 | Stage 1 trained on 8,173 crops; everything else uses 9,357 | **1. Confirmed problem** | A2 |
| 3 | The reported loss is 95 % target entropy; `KL(q‖p)` is never logged | **1. Confirmed problem** (diagnostic defect) | A4 |
| 4 | `match_view_lowpass` is documented as a Hydra override but is absent from the config group | **4. Engineering issue** (the documented experiment cannot be run as written) | A3 |
| 5 | `data_wait_fraction = 0.916`, `num_workers=0` on a 48-core host | **4. Engineering issue** | A5 / B3 / G1 |
| 6 | `data_wait_fraction` is being read as "GPU idle fraction" | **4. Engineering issue** (it is the loop-blocked fraction; it upper-bounds idleness) | A5 |
| 7 | Stage 2 reads `layers.3` post-norm, the stage DINO rewrote (CKA 0.10) and the *worst* readout of the four | **6. Genuine opportunity**, quantified at +3.3 pp linear / +3.9 pp MLP | B1 |
| 8 | SwinV2-Small vs Tiny | **3. Already ruled out** — Tiny is −0.3 pp pooled and **+0.7 pp at stage 3** for half the FLOPs. Small is dominated | B2 |
| 8b | The **initialisation** was never treated as a variable | **6. Genuine opportunity, and the largest measured one** — a frozen IN-22k trunk beats the shipped DINO encoder by **+2.24 pp** pooled with no in-domain training | B2 / App. 2 |
| 9 | The loss plateaued at epoch 20 and epochs 25/50/100 probe identically | **3. Already ruled out** — more epochs of *this* recipe will not help | D1 |
| 10 | The EMA teacher scores +0.01 pp | **3. Already ruled out** | D2 |
| 11 | 63 % of prototypes active, perplexity 772/2048 | **3. Already ruled out** — do not shrink `out_dim` on a utilisation argument | D3 |
| 12 | `same_image_minus_same_class = −0.028`; alignment worsened 0.638 → 1.111 | **1. Confirmed problem** — the objective's own invariance target was not achieved | A1 / C1 |
| 13 | Uniformity −2.15 → −3.54, participation ratio 9.1 → 18.3, k-means k=4 NMI 0.677 → 0.180 | **2. Plausible hypothesis** — a uniformity-dominated solution; KoLeo and Sinkhorn are the two candidate drivers | A1 / C3 |
| 14 | 4 of 6 views carry ~600 real pixels; 8 of 10 loss terms use one | **2. Plausible hypothesis** for the objective/data mismatch | C1 |
| 15 | Colour jitter is destroying a real cue | **3. Already ruled out as stated** — mean RGB is *more* linearly decodable after DINO (R² 0.864 → 0.908 pooled, 0.925 → 0.951 stage 3) | D5 |
| 16 | Absolute crop scale is discarded by the resize to 256 | **3. Already ruled out** — `log(area)` is 95–96 % linearly decodable from both encoders; adding it explicitly moves the probe +0.1 pp | D4 |
| 17 | Sinkhorn at `K/B_teacher = 16` imposes an entropy floor of 2.77 nats and a diffuse target | **2. Plausible hypothesis** — and the original reason for choosing Sinkhorn no longer applies | C3 |
| 18 | Five classes score F1 = 0.000; the grouped single split holds 14 of 27 classes | **5. Dataset/evaluation limitation** | E5 / E6 |
| 19 | +18.65 pp between crop-level and photograph-disjoint splitting | **5. Dataset/evaluation limitation** | E5 / D9 |
| 20 | Within one photograph, 89–98 % of crops have a neighbour above cosine 0.95 at 32×32 grey | **5. Dataset/evaluation limitation** — effective sample count ≪ 9,357 | E6 |
| 21 | `CKA(random, imagenet) = 0.335 > CKA(dino, imagenet) = 0.295` | **Not evidence of anything** — `random_init` has participation ratio 1.36 and top-1 variance share 0.851, so CKA against it is dominated by one direction | E8 |
| 22 | Stage 1 self-distils on every crop, including the photographs the evaluation holds out | **5. Dataset/evaluation limitation** — transductive; must be disclosed | E4 |
| 23 | Within-class photograph identity is decodable at +10.0 pp above chance from ImageNet features and **+3.5 pp** from the DINO features | **6. Genuine opportunity** — stage 1's one clearly beneficial effect, currently unmeasured; the instrument is missing from the evaluation | 0.7 / E9 |

---

## A. Must-fix implementation issues

### A1 — KoLeo repels the two global views of the same image

* **File.** `src/losses/dino.py:414-415` (`koleo = koleo_regularizer(student_embeddings)`), fed by `src/trainers/contrastive_pretrain.py:1514-1516` (`student_embeddings = bottleneck[: 2 * batch_size]`).
* **Current behaviour.** `bottleneck` is view-major, so rows `[0:B]` are global view 0 of every image and rows `[B:2B]` are global view 1 of the *same* images in the same order. `koleo_regularizer` takes one `[2B, 256]` block, computes the full pairwise distance matrix, masks only the diagonal, and minimises `−log(min_{j≠i} ‖z_i − z_j‖)`. Row `i` and row `B+i` are two augmented views of one crop — under a working DINO they are the *closest* pair in the block, so they are the argmin, and the term's gradient pushes them apart.
* **Proposed change.** Apply KoLeo per global view, as the DINOv2 reference does: split `student_embeddings` into `num_global_crops` chunks and sum (or mean) the per-chunk KoLeo. One expression; the weight `lambda_koleo` and everything else stays.
* **Rationale.** DINOv2's `ssl_meta_arch.py` computes `sum(self.koleo_loss(p) for p in student_cls_tokens.chunk(2))` with the in-source comment *"we don't apply koleo loss between cls tokens of a same image"*. KoLeo is a **uniformity** regulariser over *distinct instances*; applied across views it becomes an explicit **anti-alignment** term that directly opposes Eq. 1 on the one pair Eq. 1 cares most about.
* **Evidence.** `[run]` KoLeo fell from 0.369 at step 0 to ~0.03 by epoch 40 and stayed there (0.026 at the last logged step; the trajectory is noisy, not strictly monotone). Since KoLeo is `mean_i[-log d_i]`, `exp(-0.030) = 0.970` is the geometric-mean nearest-neighbour distance in the 256-D bottleneck — cosine **0.53** for the *closest* pair in a 128-row block, where a working DINO would put the two views of one crop near 0.9. `[eval]` The trunk's positive-pair cosine is 0.445 (distance 1.05) against ImageNet's 0.681, `alignment` **worsened** 0.638 → 1.111, and `same_image_minus_same_class` is −0.028. The objective's stated purpose is to raise exactly the quantity that fell.
* **Expected effect.** Higher alignment, `same_image_minus_same_class` moving toward 0 or positive, higher `self_retrieval_top1`. The reported DINO loss should fall slightly (one opposing term removed) while `KL(q‖p)` falls more.
* **Downside / failure mode.** Per-view KoLeo is a weaker uniformity force; if uniformity was carrying the small gain DINO does show, the probe could fall. Watch `spectrum/rankme` and `participation_ratio` — a collapse toward ImageNet's 9.1 with no probe gain means KoLeo was doing useful work and `lambda_koleo` should rise instead.
* **Confidence.** **High** that the current form deviates from the reference and is anti-aligned by construction. **Medium** on the size of the effect.
* **Isolate?** **Isolated.** This is the single most important arm in Phase 1 and everything else should be measured on top of the fixed version.
* **Measurement.** `invariance/alignment`, `invariance/same_image_minus_same_class`, `invariance/self_retrieval_top1`, `train/koleo`, and the headline `oof_probe_sub_accuracy_testable_classes`.

### A2 — Nothing records or checks which corpus stage 1 trained on

* **File.** `src/trainers/contrastive_pretrain.py` (startup block, ~lines 981–1021) and `src/datasets/dataset.py::get_pretrain_dataloader`.
* **Current behaviour.** The trainer logs `len(dataloader)` and the cache size but never the sample count as a checked invariant, never a content digest, and never a per-class histogram. `PretrainImageFolderDataset` is an `ImageFolder` over whatever `data.root_path` points at.
* **Proposed change.** At startup, log and write into the event stream: `len(dataset)`, the number of source groups (`source_image_id` is already implemented in `src/datasets/dataset.py:32`), the per-sub-variety count, and a stable digest of the sorted relative path list. Refuse to start (or warn loudly) if the count disagrees with `data.num_sub_varieties`-derived expectations. Have `pretrain_eval.py` compare its own dataset digest against the one in the checkpoint's run and print a prominent mismatch line.
* **Rationale.** The stage-1 → stage-2 handoff is a bare `state_dict` with no provenance. `provenance.json` already records checkpoint SHA-256, git commit and library versions — the one thing it cannot recover is *what the encoder was trained on*.
* **Evidence.** §0.1: 8,173 vs 9,357, discovered only by cross-reading two log lines against `metrics.json`. The STAGE1_EVALUATION.md report itself quotes both "8,173 images" and "9,357 crops" on the same page without reconciling them.
* **Expected effect.** No effect on representation quality; it makes every future comparison trustworthy.
* **Downside.** A digest over 9,357 paths is a few milliseconds. None.
* **Confidence.** **High.**
* **Isolate?** Not an experiment. Ship it before Phase 1 so every arm is comparable.
* **Measurement.** The mismatch line does not appear when it should not, and does when it should.

### A3 — `match_view_lowpass` is documented as a config override but is not a config key

* **File.** `conf/data/hierarchical_seeds.yaml` (`augmentation:` block); the flag exists at `src/datasets/transforms.py:174` and is honoured at `:199-206`.
* **Current behaviour.** `architecture/02_BACKBONE_AND_SSL.md` §7 and `STAGE1_EVALUATION.md` R3 both instruct `data.augmentation.match_view_lowpass=true`. The key is absent from the config group, so Hydra's struct mode rejects that override; it needs `+data.augmentation.match_view_lowpass=true`.
* **Proposed change.** Add `match_view_lowpass: false` to the `augmentation:` block, with the comment that already exists in the transform docstring. Fix the two documents.
* **Rationale.** A documented experiment that errors out is worse than an undocumented one — the error looks like a typo and gets abandoned.
* **Evidence.** `[code]` The key does not appear in `conf/`; `grep -rn match_view_lowpass conf/` returns nothing.
* **Expected effect.** None on training; it makes C2 runnable.
* **Downside.** None.
* **Confidence.** **High.**
* **Isolate?** Not an experiment.
* **Measurement.** `python main.py pretrain data.augmentation.match_view_lowpass=true experiment.training.max_batches=2` starts.

### A4 — The reported loss is not a learning curve; log `KL(teacher‖student)`

* **File.** `src/trainers/contrastive_pretrain.py:1599-1620` (`step_metrics`), and the epoch block at `:1775-1785`.
* **Current behaviour.** `train/loss` and `train/teacher_entropy` are logged separately; nothing subtracts them. Every figure, every "converged at epoch 20" statement and `stage1_dynamics.loss_*` in `metrics.json` are computed on the raw cross-entropy.
* **Proposed change.** Add `train/teacher_student_kl = loss − teacher_entropy` (both already device tensors; one subtraction, no synchronisation) and the epoch-level equivalent. Make `src/trainers/pretrain_eval.py::dynamics_panels` plot it as the primary curve and demote the raw loss to a secondary axis.
* **Rationale.** `CE(q, p) = H(q) + KL(q‖p)`. Under Sinkhorn centering, `H(q)` is set by the normaliser, `K`, `B_teacher` and the temperature schedule — none of which is the student learning. Reporting `CE` as the training curve conflates the two.
* **Evidence.** §0.2 — 80.0 % of the total loss drop is `H(q)`; the final loss is 94.8 % `H(q)`; and the KL was still improving at epoch 93 while the raw loss had been flat since epoch 20.
* **Expected effect.** No change to training. It makes "did this arm learn more?" answerable from the training log, which is the difference between screening four arms in an afternoon and running a full `eval-pretrain` on each.
* **Downside.** Under `centering="ema"` the same decomposition holds, so nothing has to change for C3.
* **Confidence.** **High.**
* **Isolate?** Not an experiment.
* **Measurement.** The new key appears in `events.jsonl` and reproduces §0.2 on the existing run.

### A5 — `data_wait_fraction` is a loop-blocked fraction, not a GPU-idle fraction

* **File.** `src/trainers/contrastive_pretrain.py:1466-1480, 1615` (the `wait_started` accumulator) and the docstring at `:72-79`.
* **Current behaviour.** `data_wait` accumulates wall clock spent inside the dataloader `__next__`. Because nothing synchronises inside the step (deliberately — see the module docstring), the queued GPU work drains *during* that window. `STAGE1_EVALUATION.md` Finding 1.3 then computes "GPU-busy time ≈ 13.34 h × (1 − 0.916) = 1.12 h", which is the CPU *enqueue* time, not the GPU busy time.
* **Proposed change.** Rename or document the metric as `loop_blocked_fraction`, and add an optional true GPU-busy measurement gated on a debug flag (`torch.cuda.Event` around the step, or `torch.cuda.synchronize()` on logging steps only). Correct the two sentences in `STAGE1_EVALUATION.md` and `architecture/02`.
* **Rationale.** The metric's *direction* is right and its operational conclusion ("the loader is the bottleneck") is right; the derived FLOP/s figure is not. The FLOP-based cross-check (415.7 PFLOPs at a plausible 50–100 TFLOP/s effective ⇒ 1.2–2.3 h) happens to land in the same range, which is why nobody noticed.
* **Evidence.** `[code]` No synchronisation exists between the backward enqueue and the next `__next__`. `[run]` `data_wait_fraction` rises 10.5 % → 84.4 % within the first 90 steps, which is the signature of a queue filling up, not of the GPU idling from step 0.
* **Expected effect.** None on training. It stops a wrong throughput number entering the paper's cost table.
* **Downside.** A synchronised measurement costs a stall; keep it off by default.
* **Confidence.** **High** on the interpretation; **medium** on the true busy fraction until measured.
* **Isolate?** Not an experiment.
* **Measurement.** `scripts/bench_pretrain_step.py --scaling 1` reports a step time; compare `steps/epoch × step_time` against the measured epoch duration.

### A6 — Stage 1 writes no `summary.json`

* **File.** `src/trainers/contrastive_pretrain.py` (final-save block, `:1925-1988`).
* **Current behaviour.** Stage 2 writes `summary.json` and `test_predictions.npz` (a documented contract in `CLAUDE.md`). Stage 1 writes checkpoints, `events.jsonl` and a budget table printed into the log. Recovering "what recipe produced this encoder" means parsing 3.7 MB of JSONL.
* **Proposed change.** Write `summary.json` next to the checkpoints: resolved augmentation node, view geometry (`transform.view_sizes`, `view_ids`), `num_crops`, effective batch, LR provenance, centering/`K`/`lambda_koleo`, the dataset digest from A2, final `KL(q‖p)`, teacher entropy and its bounds, wall clock, peak VRAM, and the SHA-256 of the published backbone.
* **Rationale.** A four-arm stage-1 comparison (Phase 1 below) is only tractable if each arm leaves one machine-readable file. `scripts/run_ablations.py` exists for stage 2 for exactly this reason and has no stage-1 counterpart.
* **Evidence.** `[code]` `grep -n "summary.json" src/trainers/contrastive_pretrain.py` → no matches.
* **Expected effect.** None on quality; it is a precondition for E7 and for the Phase-1 sequence.
* **Downside.** None.
* **Confidence.** **High.**
* **Isolate?** Not an experiment.

### A7 — The test fixture's taxonomy does not match the dataset's

* **File.** `tests/conftest.py`, `SUBVARIETY_COUNTS = {"Millet": 8, "Mustard": 3, "Rice": 13, "Seasame": 3}`.
* **Current behaviour.** The real tree is `Amaranthus: 3, Millet: 8, Mustard: 3, Rice: 13`. `sorted(SUBVARIETY_COUNTS)` yields `Millet, Mustard, Rice, Seasame`, so the synthetic parent→child map is `[0]*8 + [1]*3 + [2]*13 + [3]*3` where the real one is `[0]*3 + [1]*8 + [2]*3 + [3]*13`.
* **Proposed change.** Rename `Seasame` → `Amaranthus` and reorder, so the KL aggregation matrix `M` under test has the shape it has in production.
* **Rationale.** The hierarchy fixtures exercise a 4-parent tree with the right totals and the wrong grouping. That is exactly the class of bug the KL aggregation is most likely to have (§`CLAUDE.md`, "Aggregate the hierarchy KL in log space").
* **Evidence.** `[audit]` Directory listing of `Cropped_Samples`.
* **Confidence.** **High** that it is wrong; **low** that it is currently hiding a bug.
* **Isolate?** Not an experiment.

---

## B. High-confidence experimental changes

These are backed by measurements already in hand. I would make all four before
running a single new stage-1 arm.

### B1 — Read `layers.2` (16×16×384), not the pooled final stage

* **Files.** `src/models/builder.py:714-724` (`BackboneFeatureExtractor.forward` / `_extract`), `src/trainers/pretrain_eval.py:638-713` (evaluation), `conf/model/backbone/swinv2.yaml` (a new `feature_stage` key).
* **Current behaviour.** `_extract(return_tokens=True)` returns `backbone.forward_features(x)` — `[B, 8, 8, 768]`, post-`self.norm`; `return_tokens=False` mean-pools it. `DinoV2SwinV2Encoder` then projects 768 → 384. There is no way to read an earlier stage.
* **Proposed change.** Add `feature_stage: "final" | "stage3"` to `model.backbone`. For `stage3`, read `layers.2`'s output — timm 1.0.28 exposes `forward_intermediates(x, indices=[2], intermediates_only=True)` on `SwinTransformerV2`; a forward hook on `backbone.layers[2]` is the zero-dependency alternative. Emit `[B, 256, 384]` in grid mode, so `EmbeddingProjection` becomes 384 → 384 (Linear + LayerNorm; not an identity, because `use_norm=True`).
* **Rationale.** The last stage of a trunk specialises toward its pretraining objective. Here that is measurable and extreme: CKA 0.103 against the ImageNet initialisation at `layers.3` against 0.390 at `layers.2` (§0.3). Stage 2 is currently reading the part of the trunk that DINO consumed. `layers.2` is also natively 384 channels — the paper's `z ∈ ℝ³⁸⁴` (Eq. 4) — so the projection stops being an information bottleneck.
* **Evidence.** §0.3. Linear OOF probe **0.6111 → 0.6436 (+3.25 pp)**; 512-unit MLP OOF **0.6194 → 0.6584 (+3.90 pp)**; macro-F1 0.5630 → 0.5922. Holds for ImageNet too (0.6053 → 0.6174), so it is not an artefact of this checkpoint. Concatenating all stages (2,208-D) gives 0.6445 — no better than `layers.2` alone, so this is a *replacement*, not a multi-scale fusion.
* **Expected effect.** +2 to +4 pp on the frozen readout. On stage 2 the transfer is not guaranteed (see the caveat) but the head is fed a strictly more discriminative and better-conditioned input.
* **Downside / failure mode.** **Routing cost.** 256 tokens instead of 64 quadruples the MoE's routing slots and makes the Eq. 11 cross-attention 256×256 — 16× its current cost. Mitigation to test first: 2×2 average-pool the 16×16 grid to 8×8×384, which keeps the token budget and the identity projection. Second risk: the frozen-probe gain is measured on the *mean-pooled* stage-3 feature, and grid routing over 256 tokens is a different measurement — this is evidence for trying it, not proof the gain transfers.
* **Confidence.** **High** that stage 3 is the better frozen representation (measured twice, two head classes, group-disjoint). **Medium** that stage 2's accuracy moves by the same amount.
* **Isolate?** **Isolated**, and it needs no stage-1 retraining — run it against the existing `dinov2_swinv2_pretrained.pth`. That makes it the cheapest real gain on the table.
* **Measurement.** (i) Add `stage3` and `stage3_pooled_2x2` rows to `encoder_comparison.csv`; (ii) one stage-2 run with `model.backbone.feature_stage=stage3` against the current default, same seed, `split_protocol: grouped`, comparing sub-variety accuracy and macro-F1 and the recorded token count.

### B2 — Choose the trunk on measured transfer, not on capacity: **SwinV2-Small is the worst of the three**

* **Files.** `conf/model/backbone/swinv2.yaml` (`name`, `feature_dim`), `conf/experiment/pretrain_swinv2_dino.yaml`, `tests/conftest.py` (`SHIPPED_BACKBONE`), `tests/test_stage1_recipe.py::test_shipped_small_trunk_*`.
* **Current behaviour.** `swinv2_small_window16_256.ms_in1k`, 48.96 M parameters, 25.56 GFLOPs/view.
* **Evidence.** `[audit]` All frozen, all ImageNet-supervised, no in-domain training whatsoever; identical transform, folds and probe (Appendix 2):

  | trunk | params | GFLOPs/view | pooled 27-way | `layers.2` 27-way | pooled @ PCA-384 |
  | --- | --- | --- | --- | --- | --- |
  | `swinv2_tiny_window16_256.ms_in1k` | 27.6 M | 13.32 | 0.6021 | 0.6243 | — |
  | `swinv2_small_window16_256.ms_in1k` (current) | 49.0 M | 25.56 | 0.6053 | 0.6174 | 0.5917 |
  | **`swinv2_base_window12to16_192to256.ms_in22k_ft_in1k`** | 86.9 M | ~47 | **0.6335** | **0.6459** | **0.6228** |
  | *for reference:* the shipped `dino_epoch100` (Small, +13.34 h) | 49.0 M | 25.56 | 0.6111 | 0.6436 | 0.5978 |

* **What this says, plainly.** Changing one string in `conf/model/backbone/swinv2.yaml` to the **ImageNet-22k**-pretrained Base scores **+2.24 pp** over the shipped DINO encoder on the readout stage 2 actually uses, at **zero training cost**. The 13.34-hour stage-1 run bought +0.58 pp on the same readout. The advantage is not a dimensionality artefact: at a common PCA-384 width it is +2.50 pp.
* **Rationale.** `[eval]` Finding 2.1's decomposition is `random 0.3804 → +0.2449 ImageNet-1k → +0.0031 DINO`. Every recipe change in this document is competing for a share of that 0.31 pp term. The initialisation term is 79× larger and nobody had tried to improve it — SwinV2-Base at IN-22k→IN-1k has the same `window16_256` geometry, the same 8×8 grid, and a `feature_dim` of 1024 that `conf/experiment/pretrain_swinv2_base_dino.yaml` **already configures**.
* **The nuance that stops this being a slam dunk.** At the `layers.2` readout the two routes converge: Base IN-22k scores 0.6459 and Small+DINO scores 0.6436, and at a matched PCA-256 width they are **identical** (0.6262 vs 0.6265). Read positively, that is the best thing this audit found about stage 1: **at stage 3, 100 epochs of in-domain DINO on a Small trunk recovers what a 14×-larger pretraining corpus provides.** Read practically: B1 and B2 are largely *alternative* routes to ≈0.645, not additive ones, and whether DINO adds anything *on top of* IN-22k is exactly what Phase 1 must answer.
* **Recommendation.** Run Phase 1 on **`swinv2_base_window12to16_192to256.ms_in22k_ft_in1k`** if the compute allows it after B3 (it will: B3 alone buys ~8×, Base costs ~1.8× Small). Keep **`swinv2_tiny_window16_256`** as the cheap arm-throughput option — it is within 0.3 pp of Small pooled and **+0.7 pp** at `layers.2` for half the FLOPs, so **there is no configuration in which the current Small is the right choice.**
* **Downside / failure mode.** Base is 1.8× Small's FLOPs and 3.5× Tiny's, and `[eval]` Finding 2.11 argues extra capacity is not paying for itself. The counter is that this is not a capacity argument — Base IN-1k was not tested, so part of the gap could still be parameters rather than the IN-22k corpus. **Add `swinv2_base_window16_256.ms_in1k` to the Appendix-2 screen to separate the two**; if IN-1k Base also scores ~0.63 the effect is capacity and Tiny stands, and if it scores ~0.61 the effect is the corpus and Base IN-22k is the trunk.
* **Confidence.** **High** that Small IN-1k is dominated. **Medium-high** that the IN-22k Base is the right trunk, pending the IN-1k Base control.
* **Isolate?** A frozen-feature screen, not a training arm. Settle it in Phase 0 so every Phase-1 arm shares one trunk.
* **Measurement.** Appendix 2's table, extended with `swinv2_base_window16_256.ms_in1k`; then GFLOPs/view from the budget table on the chosen trunk.

### B3 — Give the dataloader workers, and raise the `auto` cap

* **Files.** the launch command; `src/utils/training/distributed.py:442-471` (`resolve_num_workers`, `min(per_rank_cpus, 8)`); `conf/data/hierarchical_seeds.yaml` (`num_workers: "auto"`, already correct).
* **Current behaviour.** The 13.34 h run carried `data.num_workers=0` on a host with 48 physical / 96 logical cores (`wandb-metadata.json`). Even `auto` would have capped at 8.
* **Proposed change.** Launch with an explicit `data.num_workers=16`; raise the `auto` cap to `min(per_rank_cpus, 16)` and log the resolved value (it already does). Keep `cache_images: true` — under `fork` the decoded buffer is shared copy-on-write, and `get_pretrain_dataloader` already disables the cache under `spawn`.
* **Rationale.** One sample costs six independent PIL chains. At 17.0 images/s with zero workers, the per-image CPU cost is ~59 ms; eight workers give ~136 images/s, which is still below the GPU's throughput implied by the FLOP budget (~150–200 images/s). The cap of 8 was chosen for a 4-vCPU Kaggle T4×2 and is a limiter on this machine.
* **Evidence.** `[run]` `epoch/data_wait_fraction` mean 0.916; `[run]` 48/96 cores; `[run]` peak VRAM 43.5 of 79.2 GB.
* **Expected effect.** 13.34 h → roughly 1.5–2.5 h for the identical 100 epochs. **Zero effect on the objective** — workers change who computes an augmentation, not what it is.
* **Downside.** RAM: 16 workers under `fork` share the 91 MB cache copy-on-write; under `spawn` the cache is disabled automatically. Watch for the "Disabling the image cache" warning, which would mean the start method is not `fork`.
* **Confidence.** **High.**
* **Isolate?** Not an experiment. Do it first — it converts the whole Phase-1 sequence from a week into a day.
* **Measurement.** `epoch/data_wait_fraction` below ~0.3 in epoch 1. If it does not fall, the bottleneck is elsewhere and every cost estimate below must be redone.

### B4 — Crop the 18 unused source photographs

* **Files.** outside this repository (the cropping script that produced `Cropped_Samples`), then `SEED_DATA_ROOT`.
* **Current behaviour.** 81 of 99 raw photographs were cropped; `Cropped_Samples` holds 9,357 crops from 81 groups.
* **Proposed change.** Re-run the detector/crop step on the 18 unused photographs listed in §0.6 and republish the dataset. Re-baseline: every accuracy in the paper moves, so this must happen *before* Phase 1, not between arms.
* **Rationale.** The binding constraint is the number of *scenes*, not the number of crops. This adds 22 % more scenes at zero acquisition cost, and it concentrates on classes that currently have 2–3 photographs — including KodoMillet and LittleMillet, which are directly implicated in the Millet branch's 0.413 accuracy / 0.120 macro-F1.
* **Evidence.** §0.6 — the raw files exist and were verified present. `[eval]` `sources_per_sub_variety` shows exactly which classes are photograph-starved.
* **Expected effect.** Better group-disjoint generalisation for the nine affected classes; a more informative `StratifiedGroupKFold`; ~2,000 more unlabeled crops for stage 1. It does **not** fix the five single-photograph classes.
* **Downside / failure mode.** The unused photographs may have been excluded deliberately (blur, exposure, a different tray). **Check that before cropping** — an out-of-focus photograph added to the SSL corpus is worse than nothing. Also: all published numbers move, so the paper must be re-run end to end.
* **Confidence.** **Medium-high** on the benefit; **low** until the reason for the original exclusion is established.
* **Isolate?** A corpus change, not an arm. Do it once, before Phase 1.
* **Measurement.** `group_report()` reports `num_source_groups = 99`; per-class `sources_per_sub_variety` for the nine classes; the OOF fold composition (`sub_varieties_missing_from_test` should shrink).

### B5 — Drop the input resolution to 192 px if the IN-22k trunk is adopted

* **Files.** `conf/data/hierarchical_seeds.yaml` (`image_size`, `local_crop_size`), `conf/model/backbone/swinv2.yaml` (`name`).
* **Current behaviour.** `image_size: 256` for every view, on crops with a median of 52×51 native pixels — a median 5.7× upsample for globals and 10.5× for locals (§0.5).
* **Proposed change.** With `swinv2_base_window12_192.ms_in22k`, run the whole pipeline at 192 px.
* **Rationale.** The data has no content above ~52 px per side, so the extra output resolution is interpolation that costs quadratic attention and linear patch-embed work.
* **Evidence.** `[audit]` Appendix 2 — pure IN-22k at **192 px** scores pooled **0.6342** and `layers.2` **0.6436**, against IN-22k→1k at 256 px scoring 0.6335 / 0.6459. Statistically indistinguishable, for **1.78× fewer FLOPs per view**.
* **Expected effect.** ~1.8× more stage-1 arms per unit of compute, no measured loss of transfer.
* **Downside / failure mode.** The token grid becomes 6×6 (36 tokens) rather than 8×8 (64), and `layers.2` becomes 12×12 rather than 16×16. Stage 2's routing-slot count changes with it, which matters for the load-balancing statistic's estimability at small batch (`CLAUDE.md`: grid routing is what makes it estimable at batch 16). `tests/conftest.py::PAPER_TOKEN_GRID = 64` will need to become a per-trunk constant.
* **Confidence.** **Medium-high** for the frozen transfer (measured); **medium** that it survives the token-grid change downstream.
* **Isolate?** Bundle it with the trunk decision in Phase 0, not as a separate arm.
* **Measurement.** Appendix 2 row; then the stage-2 routing statistics (`dead_experts`, `expert_label_nmi`) on the smaller grid.

---

## C. Hypotheses requiring controlled ablation

Each is one arm. All of them assume A1 is already fixed, because A1 changes the
quantity most of them are trying to move.

### C1 — Give local views enough real content to be the same object

* **Files.** `conf/data/hierarchical_seeds.yaml` (`augmentation.local_crops_scale`, `augmentation.global_crops_scale`, `data.local_crop_size`).
* **Current behaviour.** `local_crops_scale: [0.05, 0.4]` at `local_crop_size: 101`; `global_crops_scale: [0.4, 1.0]` at 256.
* **Proposed change.** One arm at `local_crops_scale=[0.30, 0.70]`, `local_crop_size=160`, `global_crops_scale=[0.70, 1.0]`.
* **Rationale.** `RandomResizedCrop` samples a fraction of the *source crop's* area, and the source crop is a single seed at a median 52×51 px. Eq. 1 asks the student to map a 24×24 px fragment and a 45×45 px view of the same seed to the same prototype distribution. For a fine-grained task whose discriminative signal is grain texture and shape, those two views frequently are not the same object in any usable sense.
* **Evidence.** §0.5 — local views carry a median **598** native pixels (p5 = 130) inflated to 65,536; **8 of the 10 cross-view terms** involve one. `[eval]` `same_image_minus_same_class = −0.028` and `alignment` 0.638 → 1.111 say the requested invariance was not achieved. The proposed setting brings the local upsample factor from 10.5× to 6.7× against the globals' 5.1×, which also shrinks the resolution shortcut C2 attacks.
* **Expected effect.** A **higher** reported DINO loss (the pretext task gets easier per-view but the targets get more informative — watch `KL(q‖p)`, which is the honest curve after A4), a less negative `same_image_minus_same_class`, and a better probe.
* **Downside / failure mode.** Narrower crop ranges reduce augmentation diversity, and with only 81 scenes the run may overfit the pretext task without learning anything transferable. The counter-signal is `rankme` / `participation_ratio` falling sharply together with `KL(q‖p)` — that is memorisation, not learning.
* **Confidence.** **Medium-high** that the current setting is wrong for this data; **medium** that the proposed one is right. Three knobs move together here; if the arm wins, split it (`local_crops_scale` alone, then `local_crop_size` alone).
* **Isolate?** **Isolated arm**, then a follow-up split if it wins.
* **Measurement.** `invariance/*`, `train/teacher_student_kl`, headline OOF probe. Also add a low-shot curve, since `[eval]` Finding 2.5 shows ImageNet currently *beats* DINO at 1/2/5 labels per class — an arm that fixes the invariance should close that.

### C2 — `match_view_lowpass=true`

* **Files.** `conf/data/hierarchical_seeds.yaml` (after A3), `src/datasets/transforms.py:199-206`.
* **Current behaviour.** Off. Global views are upsampled a median 5.7×; local views 10.5× via a 101 px intermediate. The blur signature therefore *identifies* a view as local.
* **Proposed change.** One arm with it on, everything else at the C1 baseline.
* **Rationale.** The student can satisfy part of Eq. 1 by detecting "this is a local view" and predicting the corresponding marginal, which is a shortcut that substitutes for local-to-global correspondence. The mitigation puts the globals through the same downsample-then-upsample cycle so the artefact carries no discriminative signal — as the transform's own docstring says.
* **Evidence.** `[audit]` §0.5's upsample-factor gap (10.5× vs 5.7×, p95 22.1× vs 9.9×) is the size of the cue. `[code]` The per-view blur probabilities (1.0 / 0.1 / 0.5) add a second identifying signal.
* **Expected effect.** Higher `KL(q‖p)` (the shortcut is removed), and — if the shortcut was load-bearing — a better probe.
* **Downside.** It low-passes the *global* views too, on data that already has almost no high-frequency content. This could cost more than the shortcut does. **This is the arm most likely to hurt**, and it is worth running precisely because the argument for it is theoretical and the argument against it is measurable.
* **Confidence.** **Low-medium.**
* **Isolate?** **Isolated.** It interacts strongly with C1 (which also narrows the upsample gap); if C1 wins, re-test C2 on top of C1 rather than assuming it composes.
* **Measurement.** Headline OOF probe; `invariance/self_retrieval_top1`.

### C3 — Reconsider Sinkhorn centering now that `K = 2048` and `center_momentum = 0.99`

* **Files.** `conf/model/loss/dino.yaml` (`centering`, `center_momentum`, `sinkhorn_iterations`), `src/losses/dino.py:155-250, 447-458`.
* **Current behaviour.** `centering: "sinkhorn"`, 3 iterations, `K = 2048`, `B_teacher = 128`.
* **Proposed change.** One arm at `centering="ema"` with `center_momentum=0.99`, everything else fixed.
* **Rationale — and it is specific to this repository.** Sinkhorn was adopted because the EMA centre was noise-dominated: `C ∈ ℝ^65536` estimated at `m = 0.9` from 32 vectors/step is 0.005 samples per dimension (module docstring, `src/losses/dino.py:36-39`). **That argument no longer holds.** After the revision, `C ∈ ℝ^2048`, `m = 0.99` (≈100-step window) and 128 vectors/step ⇒ **6.25 samples per dimension — 20× better than DINO's own ImageNet setting of 0.31.** What remains is Sinkhorn's cost: a doubly-stochastic assignment over `K = 2048` with `B_teacher = 128` forces every teacher row onto ≥ `K/B = 16` prototypes, an entropy floor of `log 16 = 2.77` nats, and the run sat at 5.37 nats — an effective support of ~215 prototypes per row. A target spread over 215 prototypes carries little instance identity for the student to match.
* **Evidence.** `[run]` `teacher_entropy` 5.374, floor 2.773, ceiling 7.625; `prototype_utilization` 0.991 throughout (Sinkhorn forcing it by construction). `[audit]` §0.2 — the final loss is 94.8 % target entropy, and 80 % of the "improvement" was that entropy falling. `[code]` The docstring's own justification, read against the current config values.
* **Expected effect.** Sharper targets, a larger `KL(q‖p)` to close, and — the hypothesis — more instance-discriminative pressure on the trunk. `H(teacher)` will drop and is no longer floored, so §A4's decomposition becomes essential to read the arm at all.
* **Downside / failure mode.** EMA centering is the collapse-prone path, which is why it was replaced. Watch `prototype_perplexity` and `prototype_kl_to_uniform` — a perplexity collapsing from ~2,030 toward tens is the failure, and `tests/test_losses.py` already pins the diagnostics.
* **Confidence.** **Medium.** The observation that Sinkhorn's justification expired is **high** confidence; that EMA is better here is a genuine hypothesis.
* **Isolate?** **Isolated.** Do not simultaneously change `K` or the batch — all three move the same floor.
* **Measurement.** `train/teacher_entropy`, `train/prototype_perplexity`, `train/teacher_student_kl`, headline OOF probe. A companion cheap arm: keep Sinkhorn and raise `sinkhorn_iterations` to 10, which tests whether "3 iterations does not converge" (documented in `entropy_bounds`) is contributing.

### C4 — Attach the stage-1 objective to `layers.2` as well

* **Files.** `src/models/backbones/swinv2_dino.py` (`DINO._pool`, `forward_student_views`, `DINOHead`), `conf/model/head/dino.yaml`.
* **Current behaviour.** The DINO head consumes `backbone(x)` = final norm + global average pool of `layers.3`. `layers.3` and `norm` therefore have every incentive to become part of the projection head.
* **Proposed change.** Add a second DINO head on the mean-pooled `layers.2` output (384-D input, its own prototypes), and sum the two losses with a configurable weight. Publish the trunk as before; stage 2 reads `layers.2` per B1.
* **Rationale.** B1 fixes *where stage 2 reads*. It does not fix the fact that the stage-1 gradient reaches `layers.2` only through two blocks that are being optimised for something else. If the objective supervises `layers.2` directly, the layer that ships is the layer that was trained.
* **Evidence.** §0.3 — CKA 0.103 at `layers.3` vs 0.390 at `layers.2`; DINO's gain is +2.62 pp at `layers.2` and +0.58 pp pooled. `[eval]` Finding 2.6 — the head's own 256-D bottleneck is better clustered (silhouette +0.172) than the 768-D trunk output (+0.115), i.e. the head is where the taxonomy is being learned.
* **Expected effect.** Larger DINO gain at `layers.2`; possibly a smaller one at `layers.3`, which no longer matters if B1 lands.
* **Downside / failure mode.** Two heads is more parameters, more prototypes to keep alive, and an extra weight to tune. A multi-level objective can also make the shallower stage merely mimic the deeper one. If `layers.2`'s probe does not move, the answer is no.
* **Confidence.** **Medium.**
* **Isolate?** **Isolated**, and only after B1 has been evaluated — otherwise the two effects are confounded.
* **Measurement.** `layerwise_probe/stage3` for the new encoder vs the A1 baseline; and `cka_vs_imagenet_init` per stage, which should rise at `layers.3` and fall at `layers.2`.

### C5 — Colour and grayscale strength

* **Files.** `conf/data/hierarchical_seeds.yaml` (`color_jitter_prob`, `color_jitter_saturation`, `color_jitter_hue`, `grayscale_prob`).
* **Current behaviour.** `color_jitter_prob: 0.8` (brightness/contrast ±0.4, saturation ±0.2, hue ±0.1), `grayscale_prob: 0.2`.
* **Proposed change.** One arm at `color_jitter_prob=0.3`, `grayscale_prob=0.05`.
* **Rationale, and the honest version of it.** Colour genuinely is a major class cue here — mean RGB alone scores **0.3169** on the 27-way OOF task, and mean RGB explains 78.5 % of the between-sub-variety variance `[audit]`. Only ~26.6 % of within-class colour variance is photograph-specific `[audit]`, so most of it transfers. The standard SSL argument for aggressive colour jitter is that colour histograms are a shortcut; on this dataset the "shortcut" is a large part of the signal.
* **Counter-evidence that must be stated.** The obvious inference — "colour jitter destroyed the colour cue" — is **false**. `log`-linear decodability of mean RGB from the frozen features is *higher* after DINO than before: pooled R² 0.864 → 0.908, `layers.2` R² 0.925 → 0.951 `[audit]`. Colour survived. So this arm is not fixing damage; it is asking whether capacity spent on colour invariance could be spent elsewhere.
* **Expected effect.** Small. Judge it on the *per-class* F1 of the classes that currently confuse (Jagnath ↔ Poosa33 at 0.53/0.52, AMT-4 → AMT-2 at 0.38), not on the mean — a hue cue that matters for three classes is invisible in a 27-class average.
* **Downside.** Weaker augmentation on 81 scenes risks pretext-task memorisation.
* **Confidence.** **Low-medium**, and deliberately demoted from where the existing evaluation put it (R7), because the mechanism it assumed is refuted.
* **Isolate?** Isolated, and low priority — run it only if Phase 1 leaves budget.
* **Measurement.** `tables/per_class_sub_variety.csv` for the six confusing classes; the headline probe as a guard.

### C6 — Re-decide the epoch budget on the KL curve, not the loss curve

* **Files.** `conf/experiment/pretrain_swinv2_dino.yaml` (`epochs`, `save_epochs`).
* **Current behaviour.** `epochs: 100`, `save_epochs: [25, 50, 100]`.
* **Proposed change.** Keep 100 for the Phase-1 arms with `save_epochs: [10, 25, 50, 100]`, and decide the final budget from the milestone probes *of the winning arm*, not from the current run's.
* **Rationale.** `[eval]` R2 recommends dropping to 50 epochs on the strength of the loss plateau. §0.2 shows the loss plateau is 80 % the teacher's entropy, so that argument does not support the conclusion — but the *milestone probes* (0.6276 / 0.6358 / 0.6284 at 25 / 50 / 100) do, for the current recipe. After A1 and C1 the recipe is different and the milestone curve has to be re-measured.
* **Evidence.** §0.2 and `[eval]` Finding 2.3.
* **Expected effect.** Either a confirmed 50-epoch budget (halving every future arm) or the discovery that a fixed objective keeps improving.
* **Downside.** Four milestones per arm is four extra 196 MB encoders per arm. At `keep_last_n_checkpoints: 1` the milestones are already exempt from pruning by design.
* **Confidence.** **High** that the current justification is unsound; **unknown** what the right budget is.
* **Isolate?** Free — it rides on every arm.
* **Measurement.** `tables/milestone_progression.csv` per arm.

### C7 — Preserve more of the ImageNet solution: lower LR or layer-wise decay

* **Files.** `conf/experiment/pretrain_swinv2_dino.yaml` (`lr_base`, `lr_scaling`), `src/trainers/contrastive_pretrain.py:370-432` (`resolve_learning_rate`), `:478-523` (`build_param_groups`).
* **Current behaviour.** `lr = 0.0005 × effective_batch / 256`, i.e. 1.25e-4 at batch 64, applied uniformly to every trunk parameter. This is faithful to DINO's `main_dino.py` — for a trunk trained **from scratch**.
* **Proposed change.** One arm at `lr_base=0.00025` (half), or one arm with layer-wise LR decay (0.65–0.75 per stage, lowest at `patch_embed`).
* **Rationale.** DINO's linear scaling rule is calibrated for random initialisation. Here the initialisation supplies +24.5 pp of the encoder's total worth and self-distillation supplies +0.3 pp; the optimiser has far more to lose than to gain. `[run]` `grad_norm/student_backbone/patch_embed/proj/weight` averages 1.0–1.75 and is the largest single contributor to a total gradient norm of ~1.6 — the first convolution, whose ImageNet filters are the most transferable thing in the trunk, is receiving the most gradient.
* **Evidence.** `[run]` per-tensor gradient norms; `[eval]` the +24.5 / +0.3 decomposition; §0.3's CKA 0.103 at `layers.3`.
* **Caveat that keeps this honest.** CKA is stable at `layers.0` (0.976) and `layers.1` (0.960), so the low-level filters did **not** in fact drift — the gradient norm is not evidence of movement under Adam, which normalises by the second moment. The real argument is the `layers.3` rewrite, and a lower LR attacks that only indirectly. Layer-wise decay attacks it in the wrong direction (it lowers the LR of *early* layers, which are already stable). **The better-targeted version of this idea is C4**, which changes what the deep layers are optimised *for*.
* **Expected effect.** Higher CKA against ImageNet at every stage; smaller `KL(q‖p)` improvement.
* **Downside.** Under-adapting: the whole point of stage 1 is to move the encoder. `[eval]` shows the current run moved it a lot and gained little; a lower LR could move it less and gain less.
* **Confidence.** **Low.** Listed because it is the obvious hypothesis and because the audit found the evidence usually cited for it (the patch-embed gradient) does not support it.
* **Isolate?** Isolated, low priority.
* **Measurement.** `cka_vs_imagenet_init` per stage; headline OOF probe.

### C8 — KoLeo on the trunk feature, and its weight

* **Files.** `src/trainers/contrastive_pretrain.py:1507-1516`, `src/losses/dino.py:414-418`, `conf/model/loss/dino.yaml` (`lambda_koleo`).
* **Current behaviour.** KoLeo consumes the DINO head's L2-normalised 256-D **bottleneck**. Only `student_backbone` ships to stage 2; the bottleneck is discarded.
* **Proposed change.** After A1, one arm applying KoLeo to the pooled 768-D backbone feature of each global view; and a small sweep on `lambda_koleo ∈ {0, 0.1, 0.3}`.
* **Rationale.** Uniformity is being enforced in a space nothing downstream reads. DINOv2 applies KoLeo to `x_norm_clstoken` — the backbone output — for the same reason.
* **Evidence.** `[eval]` Finding 2.6 — the bottleneck's silhouette (+0.172) is better than the trunk's (+0.115), which is consistent with the regulariser doing its work where it is applied. `[eval]` the trunk's uniformity did move (−2.15 → −3.54), so the effect propagates, but indirectly.
* **Expected effect.** Direct control of the shipped space's uniformity. The `lambda_koleo=0` arm is the clean test of whether KoLeo is worth anything at all here after A1.
* **Downside.** The 768-D trunk feature is not L2-normalised by the model; `koleo_regularizer` normalises internally, so this is safe, but the geometry it regularises is then the direction only, not the norm.
* **Confidence.** **Medium** for the space change; **high** that `lambda_koleo=0` is a necessary control.
* **Isolate?** Isolated; fold the `lambda_koleo=0` control into Phase 1 as a cheap fourth arm.
* **Measurement.** `spectrum/rankme`, `participation_ratio`, `invariance/uniformity`, headline probe.

### C9 — Raise the physical batch into the idle VRAM

* **Files.** `conf/experiment/pretrain_swinv2_dino.yaml` (`data.batch_size`, `effective_batch_size` — must move **together**), `tests/test_configs.py::test_physical_batch_is_preferred_to_accumulation`.
* **Current behaviour.** Physical batch 64, accumulation 1, peak 43.5 GB of 79.2 GB.
* **Proposed change.** `scripts/bench_pretrain_step.py --find-batch-size 64,96,112,128`, then set both keys to the largest safe value. On Tiny (B2) the headroom is much larger again.
* **Rationale.** The benefit is **statistical, not throughput**. Sinkhorn and KoLeo are per-micro-batch estimates. `B_teacher` 128 → 224 lowers the structural entropy floor `log(K/B_teacher)` from 2.77 to 2.21 nats, opening ~8 % more of the nominal range, and gives KoLeo more neighbours to be uniform against.
* **Evidence.** `[run]` peak 43.5/79.2 GB; `[code]` `entropy_bounds` and the `CLAUDE.md` rule that physical batch is not interchangeable with accumulation.
* **Expected effect.** Modest. The LR follows automatically through `resolve_learning_rate`, which is the point of deriving it.
* **Downside / failure mode.** It changes the LR and the entropy floor at once, so it is **not** a clean arm to run alongside C3. `tests/test_configs.py` asserts `data.batch_size == 32` and will need updating.
* **Confidence.** **Medium-low** on the size of the gain; **high** that it is safe.
* **Isolate?** Set it once, before Phase 1, together with B2 — or leave it alone entirely. Do not run it as an arm against C3.
* **Measurement.** `train/teacher_entropy` against its logged floor; `budget/peak_allocated_gb`.

---

## D. Changes that should NOT be made on current evidence

| # | Do not | Because |
| --- | --- | --- |
| **D1** | Add epochs to the current recipe (Table 1's 300) | `[eval]` epoch 25/50/100 probes are 0.6276 / 0.6358 / 0.6284 against a fold SD of ±0.10; the loss minimum is at epoch 90 of 100. §0.2 shows the student *did* keep improving on the pretext task for 80 more epochs and the representation did not — which is a stronger argument against, not for. |
| **D2** | Ship the EMA teacher instead of the student | `[eval]` +0.01 pp probe, +0.25 pp k-NN, *worse* k-means NMI (0.470 vs 0.490). At `momentum_teacher: 0.996 → 1.0` the two networks have met. Keep `save_teacher_in_checkpoints=true` so the check stays cheap. |
| **D3** | Shrink `out_dim` on a "prototypes are dead" argument | `[eval]` 1,291 of 2,048 prototypes win ≥1 argmax, usage entropy 0.872 normalised, perplexity 772, largest single share 0.91 %. The head is being used. (`K` is still coupled to `B_teacher` through the entropy floor — that is C3/C9's axis, and it is a *different* argument.) |
| **D4** | Feed absolute crop size / aspect ratio to the model as an explicit feature | `[audit]` `log(crop area)` is already **95.4 %** (DINO) and **96.4 %** (ImageNet) linearly decodable from the pooled features, and 95.8 % / 97.8 % at `layers.2`. Appending `[log area, log AR]` to the probe moves 27-way OOF from 0.6111 to 0.6121 (+0.10 pp). The resize does not destroy scale; the interpolation signature carries it. |
| **D5** | Reduce colour jitter on the grounds that it *destroyed* the colour cue | `[audit]` Mean RGB is **more** decodable after DINO: pooled R² 0.864 → 0.908, `layers.2` 0.925 → 0.951. The cue survived 100 epochs of `color_jitter_prob=0.8` plus 20 % grayscale. A colour arm is still worth running (C5) but on a different rationale. |
| **D6** | Move to a bigger trunk *for capacity* | `[audit]` Tiny ≈ Small on the frozen readout (−0.32 pp pooled, **+0.69 pp** at `layers.2`) at half the FLOPs. `[eval]` Finding 2.11 agrees. Capacity is not the binding constraint; **scene diversity** is. B2's IN-22k Base recommendation is *not* this — it is about the pretraining **corpus**, and B2's missing `swinv2_base_window16_256.ms_in1k` control exists precisely to keep the two apart. Do not adopt Base until that control has run. |
| **D7** | Add an iBOT / masked-patch objective to "make it DINOv2" | The patches would be predicting interpolation. §0.5: a global view carries a median 2,035 real pixels rendered into 65,536; at the `layers.0` grid each of the 4,096 tokens covers 4×4 output pixels, which at the median 5.7× upsample is **~0.5 native pixels**. A masked-patch objective on this data reconstructs the bicubic kernel. |
| **D8** | Optimise for a lower DINO loss | §0.2 — the reported loss is 94.8 % target entropy. It can be lowered by making the teacher sharper, which is not learning. After A4 the honest quantity is `KL(q‖p)`, and even that improved 2× over epochs 20–100 with no representation gain. |
| **D9** | Fall back to `split_protocol: stratified` because the numbers look better | `[eval]` +18.65 pp on the 27-way probe and +17.92 pp on k-NN between crop-level and photograph-disjoint splitting; retrieval P@1 falls 0.664 → 0.475 once same-photograph neighbours are excluded. `[audit]` 89–98 % of crops in a photograph have a neighbour above cosine 0.95 at 32×32 grey. That gap is near-duplicate matching. |
| **D10** | Turn on `model.loss.distributed_sinkhorn=true` as a free improvement when running on 2 GPUs | It is exact but it is a **different objective** — the assignment becomes doubly stochastic over the concatenated global batch. Defensible, and not comparable to the single-GPU arms. Keep it out of Phase 1. |
| **D11** | "Fix" the five zero-F1 classes with a protocol change | `[eval]` Baryard, Browntop, FingerMillet, PearlMillet, ProsaMillet have crops from exactly one photograph — `[audit]` confirmed, and `RAW_Samples` holds no second photograph for any of them. Under any photograph-disjoint split they are unpredictable, identically for every encoder including random. The fix is a camera (E6), not a splitter. |
| **D12** | Read `CKA(random, imagenet) = 0.335 > CKA(dino, imagenet) = 0.295` as "DINO drifted further than random" | `random_init` has participation ratio **1.36** and top-1 variance share **0.851** `[audit]`. Linear CKA against a near-rank-1 reference measures the shared dominant direction, not similarity of representation. |

---

## E. Dataset and evaluation changes

### E1 — Add a trivial-feature floor to `encoder_comparison.csv`

* **File.** `src/trainers/pretrain_eval.py` (`resolve_encoder_specs` / `evaluate_encoder`), `conf/experiment/eval_pretrain_representation.yaml` (`encoders:`).
* **Change.** A `handcrafted_floor` pseudo-encoder: `[log(w·h), log(w/h), mean RGB, std RGB, mean grey, std grey]`, scored by the identical probe and folds.
* **Rationale.** `random_init` (0.3804) fixes the floor a linear readout gets from architecture alone. It does **not** answer "how much does a deep encoder add over trivial image statistics?", which is the question a reviewer of a fine-grained-classification paper asks first.
* **Evidence.** §0.4 — 10 numbers score **0.5360 / macro-F1 0.4899**, 15.6 pp *above* the untrained 48.96 M trunk and 9.2 pp below the shipped one.
* **Confidence.** **High.** This is a reporting obligation, not an optimisation.
* **Measurement.** The row appears; the delta to `dino_epoch100` is stated in the paper.

### E2 — Record and cross-check the pretraining corpus

Pairs with A2. `provenance.json` gains the pretraining dataset digest read from
the checkpoint's run, and the report states plainly when it differs from the
corpus being evaluated. Confidence **high**.

### E3 — Promote the stage-3 readout to a headline row

`layerwise_probe` currently lives in one figure (`fig15`) and one CSV. Given
§0.3, `oof_probe_sub_accuracy` should be reported at both `pooled` and
`layers.2` for every encoder, and the layerwise probe should run for **all**
encoders (`experiment.evaluation.layerwise.encoders` currently lists only
`dino_epoch100` and `imagenet_init`, and `dino_epoch25/50` caches carry no stage
features at all — visible in `features/*.npz`). Confidence **high**.

### E4 — Disclose that stage 1 is transductive

`get_pretrain_dataloader(data_dir=cfg.data.root_path, ...)` self-distils on
**every** crop, including the photographs the evaluation and stage 2 hold out.
The labels never leak, and this is a normal in-domain-SSL setting, but a paper
claiming photograph-disjoint generalisation must say so. Optional control: an
arm pretrained only on the training-fold photographs. Note the direction — the
current setup favours DINO, and DINO still did not win, so the negative result is
conservative. Confidence **high** for the disclosure; **low** priority for the
control.

### E5 — Endorse `grouped_cv` for stage 2

`[eval]` R9. The grouped `GroupShuffleSplit` test split holds **14 of 27**
sub-varieties (`metrics.json → split.sub_varieties_missing_from_test`, 13 names),
so a stage-2 macro-F1 on it is mechanically capped near 14/27 for reasons
unrelated to the model. Add `grouped_cv` to
`src/trainers/moe_finetune.py::split_dataset` using `StratifiedGroupKFold` and
report out-of-fold metrics — the same thing `grouped_cv_readout` already does in
`pretrain_eval.py`. Until then, report the number of classes present in the test
split next to every macro-F1. Confidence **high**.

### E6 — Photography, in priority order

1. **Crop the 18 existing unused photographs** (B4) — free, +22 % scenes, helps 9 classes.
2. **Photograph the five single-photograph varieties** in ≥2 more sessions each — the *only* thing that makes Baryard, Browntop, FingerMillet, PearlMillet and ProsaMillet (1,387 crops, 14.8 % of the dataset) testable at all.
3. Then everything else to ≥3 sessions.

Within-photograph crop redundancy is high (`[audit]` 89–98 % of crops have a
neighbour above cosine 0.95 at 32×32 grey), so more crops per photograph is worth
much less than more photographs. Confidence **high**.

### E7 — Stage-1 dynamics panel additions

`dynamics_panels` should plot `KL(q‖p)` (A4) as the primary curve and add
`invariance/same_image_minus_same_class` and `invariance/alignment` to the
per-encoder summary table, since those are the quantities A1 and C1 are steering.
Confidence **high**.

### E8 — Two interpretation caveats to add to `STAGE1_EVALUATION.md`

* CKA against `random_init` is not comparable (D12) — the report already carries an analogous caution for `fisher_ratio` and `calinski_harabasz`; this needs the same treatment.
* "GPU was idle for 91.6 % of the run" should become "the training loop spent 91.6 % of its wall clock blocked in the dataloader", with the derived TFLOP/s figure either removed or re-derived from a synchronised measurement (A5).

---

### E9 — Add nuisance decodability as a first-class metric

* **File.** `src/utils/representation.py` (a new function), `src/trainers/pretrain_eval.py::evaluate_encoder`.
* **Change.** Report **within-sub-variety photograph identity** — nearest-class-mean or a linear probe predicting `source_groups` restricted to one sub-variety at a time, averaged over the 22 classes with ≥2 photographs, alongside the per-class chance level. `source_groups` is already carried in `features/*.npz` and `dataset.source_groups()` already exists.
* **Rationale.** Every other metric in the evaluation asks "how much *signal* is in the representation". None asks "how much *photograph nuisance* is in it", which is precisely what the grouped protocol punishes and what an in-domain SSL stage should remove. Reporting only the readout hides stage 1's one demonstrated success.
* **Evidence.** §0.7 — ImageNet +10.0 pp above chance, DINO +3.5 pp. A 65 % reduction that no existing figure or table shows.
* **Expected effect.** None on training. It gives every Phase-1 arm a second, independent axis to be judged on, and it is the mandatory falsifier for F1.
* **Downside.** Low nuisance decodability is not automatically good — an encoder that discards everything scores 0. Read it *jointly* with the sub-variety readout, the way the report already reads entropy jointly with its bounds.
* **Confidence.** **High.**
* **Isolate?** Not an experiment.

## F. Optional architectural experiments

### F1 — Provenance-derived positives: two crops from the same photograph

* **Files.** `src/datasets/dataset.py` (a sampler that pairs crops sharing `source_image_id`), `src/trainers/contrastive_pretrain.py` (view assembly).
* **Idea.** Replace some or all of the four local views with *other crops from the same source photograph*. Provenance is already parsed (`SOURCE_IMAGE_PATTERN`, `source_image_id`), so this needs no labels.
* **Rationale.** DINO's positives are augmented views of one crop, and the invariance it teaches is to crop/blur/colour. The invariance the downstream task needs is to *which individual seed* — and two crops from one photograph are, by construction, two individuals of the same variety. This is the one idea in this document that changes *what invariance is being learned* rather than how hard it is to learn.
* **Evidence.** `[eval]` `same_image_minus_same_class = −0.028`: augmented views of one crop are already *less* similar than two crops of one variety, i.e. the pretext positives are harder than the semantic ones and teach less. `[audit]` within-photograph crops sit at cosine 0.97 at 32×32 grey — already near-positives at low frequency, differing exactly in the high-frequency texture the task needs.
* **Downside / failure mode.** Same-photograph crops share lighting, background and sensor noise, so the model can satisfy the objective by learning **photograph identity** — which is precisely what the photograph-disjoint protocol is designed to punish. That makes it a *self-detecting* risk, and §0.7 supplies the detector: within-class photograph decodability is currently **+3.5 pp above chance** for the DINO encoder, down from ImageNet's +10.0 pp. If this arm pushes that number back up, it has learned the confound rather than the variety.
* **Confidence.** **Medium** that it is the right invariance; **medium** that the confound sinks it. Note the two effects are separable in principle: pulling *crops of the same photograph* together lowers within-photograph variance, which does not by itself raise between-photograph separation — the measurement decides.
* **Isolate?** Isolated, Phase 3. Run two sub-arms: all four local views replaced, and two of four.
* **Measurement.** In priority order: (1) **E9's within-class photograph decodability** — this is the gate; (2) grouped OOF probe **and** stratified probe, whose gap (`leakage.delta_probe_sub`, currently +18.65 pp) is the second detector; (3) the headline OOF probe.

### F2 — Keep the DINO head's bottleneck

`[eval]` R11 / Finding 2.6 — the 256-D bottleneck's sub-variety silhouette
(+0.172) beats the 768-D trunk output's (+0.115) on the same crops, with 45
directions carrying 95 % of its variance. Stage 1 discards it. Cheapest test:
probe `bottleneck → 27` directly (the head is already saved in
`dino_milestone_epoch_0100.pth`). If it beats `layers.2` pooled (0.6436), consider
initialising `DinoV2SwinV2Encoder.projection` from the head's first MLP layer.
Caveat: the bottleneck is L2-normalised and trained to serve the prototype task,
so a better silhouette does not automatically mean a better input to a 9 M-param
MoE head. Confidence **low-medium**.

### F3 — Scale-preserving input instead of resize-to-square

Feed each crop at its native pixel scale, centred in a 256×256 canvas with
reflection padding, instead of `Resize((256, 256))`. This keeps the true sensor
resolution and removes the class-correlated interpolation factor entirely.
**Note this is not D4** — D4 refutes *adding a scale feature*; this changes the
image formation. Downside: 51 % of crops occupy under 64×64 of the canvas, so
most tokens see padding, and it breaks comparability with every published number.
Confidence **low**; listed because it is the only proposal that removes the
resolution confound at its source.

### F4 — Push the input resolution below 192 px

B5 already establishes that **192 px loses nothing** (Appendix 2). Going further
needs a different family: `swinv2_*_window16_256` asserts 256 px and
`swinv2_*_window12_192` asserts 192 (`tests/test_models.py` pins that
`dynamic_img_size=True` does not lift the assertion). The `swinv2_cr_*` variants
accept arbitrary `img_size` and would allow 128 px — 4× fewer FLOPs than 256 on
data whose median content is 52×51 px, and a plausible fit given that a 128 px
view of a median crop is still a 2.5× upsample. `validate_swinv2_name` admits
them (`swinv2_cr_*` starts with `swinv2`), but they are a different architecture
with different ImageNet weights and only `sw_in1k` tags, so this is a real
experiment. Confidence **low**; potentially the largest single compute saving in
this document, and it would make Phase-1 arms nearly free.

---

## G. Compute and throughput (no scientific behaviour change)

| # | Change | Effect | Confidence |
| --- | --- | --- | --- |
| **G1** | `data.num_workers=16`; raise `resolve_num_workers`'s `auto` cap from 8 to 16 (`src/utils/training/distributed.py:466`) | ~6–9× wall clock. Identical arithmetic. | High |
| **G2** | `data.batch_size` and `effective_batch_size` raised together into the idle 45 % of VRAM | Better GPU utilisation; **also changes the objective's batch statistics** — see C9, do not treat as free | High (safety), Medium (benefit) |
| **G3** | SwinV2-Tiny (B2) | 25.56 → 13.32 GFLOPs/view | High |
| **G4** | Use both H100s: `python main.py pretrain --gpus 2` with `effective_batch_size` pinned | The run used 1 of 2 GPUs (`wandb-metadata.json: gpu_count = 2`). `resolve_accumulation` holds the effective batch, and `DINO.no_sync()` keeps the interconnect traffic at one reduction per step. Note the corollary already documented in `CLAUDE.md`: splitting the batch across ranks halves what Sinkhorn and KoLeo estimate from, so raise `data.batch_size` per rank rather than splitting. | High |
| **G5** | Already correct, do not undo: `output_uint8: true`, `defer_local_upsample: true`, `sdpa_attention: true` (48 modules converted, parity-checked), `cache_images: true`, `return_original=False`, one fused forward per view block, fused AdamW, `torch._foreach_` EMA | These are why 384 student views/step fit in 43.5 GB | High |
| **G6** | Do **not** switch `compile.mode` to `reduce-overhead` | CUDA graphs capture parameter pointers; the teacher's parameters are rewritten in place by the EMA every step | High |

---

## Interactions, and the minimal sequence that identifies causality

### Interaction map

```
B3/G1/G4 (workers, 2 GPUs) ──────────── scientifically inert, do first
B4 (re-crop 18 photos) ─────────────── changes the corpus ── re-baseline after it
B2 + B5 (trunk, IN-22k, 192 px) ─────── choose ONCE, before any arm
B1 (read layers.2) ──────────────────── no retraining; confounds C4 if simultaneous
   └── largely REDUNDANT with B2 at layers.2 (0.6436 vs 0.6459) ── expect ~one gain, not two
A1 (KoLeo per view) ─────────────────── changes alignment ── becomes the new baseline
   ├── C8 (lambda_koleo=0) ──────────── the control that says whether A1 mattered
   ├── C1 (crop scales) ─────────────── also changes alignment ── isolate from A1 and C2
   ├── C2 (match_view_lowpass) ──────── also narrows the upsample gap ── isolate from C1
   ├── C3 (ema centering) ───────────── moves H(teacher) ── isolate from C9 and from K
   └── C4 (supervise layers.2) ─────── only meaningful after B1; confounds B1 if simultaneous
C9 (physical batch) ─────────────────── moves LR *and* the entropy floor ── set once, never as an arm beside C3
C5 (colour), C7 (LR) ────────────────── low priority, run only if budget remains
```

**Four genuine confounds.** (A1, C1) and (C1, C2) both act on alignment and on
the local/global upsample gap. (C3, C9) both move the teacher's entropy floor
`log(K/B_teacher)`. (B1, C4) both concern which stage carries the signal.
And — the one that is easy to miss — **(B1, B2) are not additive**: at `layers.2`
the IN-22k Base and the DINO-trained Small score 0.6459 and 0.6436, and
0.6262 / 0.6265 at matched width. Budget for *one* gain of ≈+3 pp, not two.

### The sequence

**Phase 0 — no training. One day.**

| Step | Action | Gate |
| --- | --- | --- |
| 0.1 | A2, A3, A4, A6 (provenance, config key, KL metric, `summary.json`) | — |
| 0.2 | B3 / G1 / G4 (workers, both GPUs) | `data_wait_fraction` < 0.3 in epoch 1 of a 2-epoch smoke run |
| 0.3 | E1, E3 (floor row, stage-3 as a headline) | the floor row reproduces 0.5360 |
| 0.4 | **B1 screening**: probe `layers.2`, `layers.2` 2×2-pooled, and `pooled` for every existing encoder | confirm +3 pp; pick the readout |
| 0.5 | **B2 screening**: Appendix 2 is done except for the one control that decides it — add `swinv2_base_window16_256.ms_in1k` to separate *capacity* from *IN-22k corpus* (7 min of frozen extraction) | pick the trunk and the input resolution |
| 0.6 | B4 if the 18 photographs are usable | inspect them first |

**Phase 1 — five 50-epoch arms on the chosen trunk and corpus. One day at B3's throughput.**

| Arm | Change | Judged on |
| --- | --- | --- |
| **P1-F** | *no training at all* — the frozen chosen trunk | **the reference that decides whether stage 1 is worth running** |
| **P1-0** | baseline: current recipe on the chosen trunk and corpus | reference for every arm below |
| **P1-A** | **A1** KoLeo per view | `same_image_minus_same_class`, `alignment`, `self_retrieval_top1`, probe |
| **P1-B** | A1 + **C8** `lambda_koleo=0` | the control that says whether KoLeo helps at all |
| **P1-C** | A1 + **C1** local/global crop scales | `KL(q‖p)`, probe, low-shot curve |
| **P1-D** | A1 + **C3** `centering="ema"` | `teacher_entropy`, `prototype_perplexity`, probe |

Every arm also reports E9's within-class photograph decodability, because §0.7
shows that is the one axis on which the current recipe demonstrably works and an
arm could win the probe by re-learning the nuisance.

Every arm at `save_epochs=[10,25,50]` (C6). Compare on
`oof_probe_sub_accuracy_testable_classes` at the readout Phase 0 chose, with
`oof_probe_sub_f1_macro` and the fold SD.

**Phase 2 — resolve the survivors. One day.**

Take whichever of P1-C / P1-D won, add **C2** on top of it as one arm and **C4**
as another (C4 only if B1 landed), plus the best single arm re-run at 100 epochs.

**Phase 3 — confirm, with error bars.**

The winning configuration at 3 seeds, 100 epochs, plus the P1-0 baseline at 3
seeds. Only now is a stage-1 claim reportable: `[eval]`'s own fold SD is ±0.10 on
the 27-way probe, so a single arm cannot resolve anything below ~2 pp, and the
component effects in play are 0.5–4 pp.

**Infrastructure this sequence needs and the repo does not have.**
`scripts/run_ablations.py` covers stage-2 head variants only. Phase 1 needs a
`scripts/run_stage1_ablations.py` that runs each arm as a subprocess (Hydra
initialises once per process), points `experiment.training.save_path` and
`shared_backbone_path` at per-arm directories, then runs `eval-pretrain` into a
per-arm `experiment.evaluation.save_path` and collects the headline metrics into
one table. Without it, arms will silently overwrite each other's
`outputs/eval_pretrain/` and each other's `outputs/checkpoints/dinov2_swinv2_pretrained.pth`.

### The one question Phase 1 exists to answer

Appendix 2 shows that a frozen IN-22k trunk, with **no in-domain training at
all**, already reaches the level the 13.34-hour DINO run reached. So the question
is no longer "can stage 1 be improved" but:

> **Does self-distillation add anything on top of a strong prior, on 81 scenes?**

P1-0 (IN-22k trunk, current recipe) against the frozen IN-22k trunk answers it
directly, and it is the first number to look at. If P1-0 does not beat the frozen
trunk by more than a fold SD, stage 1 as an *objective* is not earning its
compute, and the honest paper claim changes from "in-domain SSL adapts the
encoder" to "in-domain SSL removes photograph nuisance (§0.7) without improving
the readout".

### What would falsify the whole framing

If P1-A, P1-C and P1-D all land within ±1 SD of P1-0, then the objective is not
the binding constraint. The answer is then E6 — **81 scenes is the ceiling** — and
the remaining effort belongs in data acquisition (B4, E6), in the initialisation
(B2), and in the stage-2 readout (B1): three changes that together are worth
roughly 3–5 pp and none of which is a stage-1 recipe change.

---

## Appendix 1 — Reproducing this audit's measurements

Every `[audit]` number above came from one of these, run from the repository root
with the dataset at
`../Dataset/Hierarchical_SeedData/Cropped_Samples` and the cached features in
`outputs/eval_pretrain/features/`:

1. **Crop geometry, colour and provenance** — a pass over `Cropped_Samples`
   recording `(seed_type, sub_variety, source, w, h, mean RGB, std RGB)`; then
   between-group variance shares and the `StratifiedGroupKFold` probe on the 10-D
   hand-crafted vector.
2. **Unused raw photographs** — set difference between the `RAW_Samples` stems
   and the `_bbox` prefixes in `Cropped_Samples`.
3. **Per-stage CKA and layerwise OOF probes** — `features/imagenet_init.npz` and
   `features/dino_epoch100.npz` already carry `stage_stage1..4` and `pooled`;
   linear CKA in the feature-covariance form, and
   `StratifiedGroupKFold(5, shuffle=True, random_state=42)` +
   `StandardScaler → LogisticRegression(C=100)`. The MLP probe is a 512-unit
   GELU MLP, dropout 0.2, AdamW 1e-3, cosine over 80 epochs.
4. **Scale and colour decodability** — `GroupKFold(5)` + `Ridge(alpha=10)`
   predicting `log(w·h)` and mean RGB from the frozen features; OOF R².
5. **View content** — Monte-Carlo over the measured crop areas with the
   configured `RandomResizedCrop` scale ranges.
6. **Loss decomposition** — `train/loss − train/teacher_entropy` over the 1,270
   logged steps in `outputs/hydra/2026-08-14/17-25-07/events.jsonl`.

Two sanity checks that passed and should stay passing: the label and group
vectors derived independently from the directory tree match
`features/*.npz`'s `sub_labels` / `source_groups` exactly; and
`outputs/checkpoints/dinov2_swinv2_pretrained.pth` and
`dino_backbone_epoch_0100.pth` have different SHA-256s and **all 423 tensors
bit-identical**, exactly as `compare_checkpoints` documents — so the evaluated
encoder and the shipped encoder are the same weights.

## Appendix 2 — Initialisation screen (frozen features, no in-domain training)

Every row: frozen timm trunk, `Resize((S,S), BICUBIC)` + ImageNet normalisation,
fp32, mean-pooled features, `StratifiedGroupKFold(5, shuffle=True,
random_state=42)` over all 9,357 crops with the source photograph as the group,
`StandardScaler → LogisticRegression(C=100)`. `[audit]`

| trunk (all ImageNet-supervised, **no DINO**) | px | params | pooled 27-way | macro-F1 | `layers.2` 27-way | macro-F1 | 4-way |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `swinv2_tiny_window16_256.ms_in1k` | 256 | 27.6 M | 0.6021 | 0.5537 | 0.6243 | 0.5756 | 0.9844 |
| `swinv2_small_window16_256.ms_in1k` *(current)* | 256 | 49.0 M | 0.6053 | 0.5561 | 0.6174 | 0.5708 | 0.9830 |
| `swinv2_base_window12to16_192to256.ms_in22k_ft_in1k` | 256 | 86.9 M | **0.6335** | **0.5825** | **0.6459** | **0.5949** | 0.9836 |
| `swinv2_base_window12_192.ms_in22k` | **192** | 87.9 M | **0.6342** | **0.5842** | 0.6436 | 0.5922 | 0.9835 |
| *reference:* `dino_epoch100` (Small + 13.34 h in-domain DINO) | 256 | 49.0 M | 0.6111 | 0.5630 | 0.6436 | 0.5922 | 0.9869 |

Matched-width control, to rule out "wider features probe better":

| representation | full width | PCA to a common width |
| --- | --- | --- |
| Base IN-22k→1k pooled | 0.6335 (1024) | **0.6228** (384) |
| Small IN-1k pooled | 0.6053 (768) | 0.5917 (384) |
| Small + DINO pooled | 0.6111 (768) | 0.5978 (384) |
| Base IN-22k→1k `layers.2` | 0.6459 (512) | 0.6262 (256) |
| Small IN-1k `layers.2` | 0.6174 (384) | 0.6007 (256) |
| Small + DINO `layers.2` | 0.6436 (384) | 0.6265 (256) |

**Three readings, in order of how actionable they are.**

1. **The initialisation is worth more than the stage-1 run.** IN-22k Base beats
   the shipped DINO encoder by **+2.24 pp** pooled (**+2.50 pp** at matched
   width) with no in-domain training at all, against DINO's own +0.58 pp over its
   Small IN-1k starting point.
2. **At `layers.2` the two routes converge exactly.** Base IN-22k 0.6459 vs
   Small+DINO 0.6436 — and 0.6262 vs 0.6265 at matched width. So B1 and B2 are
   substantially *alternative* routes to ≈0.645, and the open question Phase 1
   must answer is whether DINO adds anything **on top of** a strong prior.
3. **256 px is not needed.** Pure IN-22k at **192 px** matches IN-22k→1k at 256 px
   (pooled 0.6342 vs 0.6335; `layers.2` 0.6436 vs 0.6459) for **1.78× fewer
   FLOPs** — the empirical version of §0.5's argument that the data has no content
   above ~52 px. Its 6×6 grid (`layers.2`: 12×12) changes stage 2's routing token
   count, which is a real but bounded consequence.

**Missing control, and it decides B2.** `swinv2_base_window16_256.ms_in1k` was not
screened, so the Base advantage is not yet separated into *capacity* versus
*IN-22k corpus*. Run it first — it is 7 minutes of frozen extraction. If it scores
≈0.63 the effect is capacity (and Tiny is then the efficient choice, since Small
is dominated either way); if ≈0.61 the effect is the corpus, and IN-22k is the
trunk to build on.

---

# Implementation status

*Added after the plan was implemented. The document above is unchanged and remains
the specification; this section records what was built, what was deliberately not
built, and where the implementation had to depart from the plan.*

Test suite: **518 → 603 passing**. Every experiment config composes; `main.py
smoke` runs both stages end to end; `scripts/dry_run.py` is green.

## 1. Implemented

### A — must-fix implementation issues

| # | What was built | Where |
| --- | --- | --- |
| **A1** | `grouped_koleo(features, num_groups, scope, reduction)` chunks the view-major global block by view before taking nearest neighbours. `model.loss.koleo_scope` defaults to `per_view`; `all_views` restores the pre-audit behaviour **as a control**, and the trainer logs a warning when it is selected. `koleo_reduction` is `mean` (keeps `lambda_koleo` on the same scale, so the two are a single-factor comparison) or `sum` (DINOv2 literally). | `src/losses/dino.py`, `conf/model/loss/dino.yaml` |
| **A2** | `corpus_fingerprint()` — SHA-256 over the sorted list of dataset-**relative** POSIX paths, plus sample/class/source-group counts and a per-class histogram. Logged at startup, written to `events.jsonl` as a `corpus` event and to `summary.json`. `experiment.training.corpus_check` is `warn` (default) / `error` / `off`. `pretrain_eval` reads the stage-1 `summary.json` back and prints `describe_fingerprint_mismatch(...)` when the corpora differ. | `src/datasets/dataset.py`, `src/trainers/contrastive_pretrain.py`, `src/trainers/pretrain_eval.py` |
| **A3** | `data.augmentation.match_view_lowpass: false` now exists as a config key, so the documented override composes without a `+`. A test pins that the documented command works. | `conf/data/hierarchical_seeds.yaml` |
| **A4** | `compute_dino_loss` always emits `dino_cross_entropy`, `teacher_entropy_cross_view` and `teacher_student_kl` as **device tensors** — *not* gated on `metrics_enabled`, because the epoch mean has to see every micro-batch. `H` is weighted by each teacher view's share of the cross-view pairs, so the decomposition is exact for any mask. The trainer accumulates all three on-device and all-reduces the epoch means; step and epoch logs read `loss=… \| CE=… = KL … + H …`; the loss figure plots the KL first. `PretrainDynamics.summary()` gains `teacher_student_kl_{initial,final,min}` and `teacher_entropy_share_of_loss_{final,improvement}`. | `src/losses/dino.py`, `src/trainers/contrastive_pretrain.py`, `src/utils/evaluation.py` |
| **A5** | The metric is renamed `loop_blocked_fraction` (`data_wait_fraction` kept as an alias so existing figures and parsers keep working) and documented as an upper bound on idleness. `experiment.training.measure_gpu_busy=true` adds `GpuBusyMeter` — CUDA events around each micro-batch's compute, **drained in batches at logging steps** so the stall is paid once per logging interval rather than once per micro-batch. Off by default; a no-op on CPU/MPS. The two wrong sentences in `STAGE1_EVALUATION.md` and the derived TFLOP/s figure are struck through in place, with the reason. | `src/trainers/contrastive_pretrain.py`, `STAGE1_EVALUATION.md`, `README.md` |
| **A6** | `write_stage1_summary()` writes `summary.json` beside the checkpoints using the same `RunSummary` shape stage 2 uses — resolved augmentation, view geometry, effective batch, LR provenance, `criterion.loss_flags()`, the corpus fingerprint, the final `KL(q‖p)`, teacher entropy with its bounds, wall clock, peak VRAM. An **interrupted** run writes one too, flagged `completed: false`. | `src/trainers/contrastive_pretrain.py` |
| **A7** | `SUBVARIETY_COUNTS` is now `{Amaranthus: 3, Millet: 8, Mustard: 3, Rice: 13}`. The three tests that had encoded the old (wrong) ordering as literals now derive their expectations from the fixture, so the two cannot agree with each other and disagree with production again. | `tests/conftest.py`, `tests/test_losses.py`, `tests/test_metrics.py` |

### B — high-confidence experimental changes

| # | What was built |
| --- | --- |
| **B1** | `model.backbone.feature_stage`: `final` (default, unchanged) / `stage3` (`layers.2`, 16×16×384) / `stage3_pooled_2x2` (2×2-pooled back to 64 tokens). Extraction uses timm's `forward_intermediates(..., stop_early=True)`, so reading `layers.2` genuinely skips the later blocks; a hook fallback covers a timm without it, and a test pins that the two produce **bit-identical** tensors. `feature_dim` reports the *emitted* width (384), with `backbone_feature_dim` still reporting the trunk's. On the evaluation side, `experiment.evaluation.readout.stages: [pooled, stage3]` runs the headline out-of-fold probe at each stage into `tables/readout_stages.csv` and `oof_probe_sub_accuracy_at_*` columns. |
| **B2** | An encoder row may name its own `backbone` and `image_size`, which makes Appendix 2 a config file: `conf/experiment/screen_backbones.yaml` (`python main.py screen-backbones`), **including the missing `swinv2_base_window16_256.ms_in1k` control**. Trunk experiments added for Tiny and IN-22k Base, each publishing to its own `shared_backbone_path`. The evaluation builds a per-resolution loader so a 192 px row is not measured through a 256 px resize. |
| **B3 / G1** | `resolve_num_workers(auto_cap=…)`, default raised 8 → 16, exposed as `data.num_workers_auto_cap` and plumbed through both trainers. |
| **B4** | `raw_photograph_coverage()` plus `scripts/report_raw_photographs.py` and `data.raw_photographs_root`. Both trainers report uncropped photographs when the raw tree is configured. **Reporting only** — see §2. |
| **B5** | 192 px is runnable (`screen_backbones` includes `swinv2_base_window12_192.ms_in22k` at 192). `PAPER_TOKEN_GRID` became a per-trunk constant as the plan predicted: `PAPER_TOKEN_GRID = 64`, `TOKEN_GRID_192 = 36`, `STAGE3_TOKEN_GRID = 256`, `STAGE3_TOKEN_GRID_192 = 144`, all re-measured by a test. |

### C — the arms

Every arm in Part C is reachable as a Hydra override and is defined in a manifest
under `conf/stage1_arms/`, run by `scripts/run_stage1_ablations.py`.

| # | Override | Notes |
| --- | --- | --- |
| **C1** | `data.augmentation.local_crops_scale` / `global_crops_scale` / `data.local_crop_size` | already config; `P1-C` in `phase1.yaml` |
| **C2** | `data.augmentation.match_view_lowpass=true` | now a real key (A3); `P2-C2` |
| **C3** | `model.loss.centering=ema` | already config; `P1-D` |
| **C4** | `model.head.aux_stage=2`, `aux_out_dim`, `aux_weight` | **new**: a second `DINOHead` on the mean-pooled `layers.2` output, with its own prototypes, its own EMA teacher copy and its own `CustomDINOLoss`. Captured by a forward hook inside the *same* forward, not a second pass. Registered on the DDP wrapper so its gradients reduce; `student_backbone.state_dict()` is byte-identical with or without it. Not allocated at all when `aux_stage: null`. `P2-C4` |
| **C5** | `data.augmentation.color_jitter_prob` / `grayscale_prob` | already config |
| **C6** | `experiment.training.save_epochs` | already config; the Phase-1 manifest sets `[10,25,50]` |
| **C7** | `experiment.training.lr_base` | already config — see §2 for layer-wise decay |
| **C8** | `model.loss.lambda_koleo=0` and `model.loss.koleo_space=backbone` | **new** for the space: `forward_student_views(return_features=True)` returns the pooled trunk feature, and the trainer only asks for it when something will read it. `P1-B` |
| **C9** | `data.batch_size` + `effective_batch_size` | already config; deliberately **not** changed — see §2 |

### E — dataset and evaluation

| # | What was built |
| --- | --- |
| **E1** | `handcrafted_image_features()` (10 numbers, read at **native** resolution) plus `kind: handcrafted` encoder rows, cached on the corpus digest rather than on a checkpoint. Present in `eval_pretrain_representation`, `eval_frozen_reference` and `screen_backbones`. |
| **E2** | The corpus fingerprint appears in `provenance.json` (`corpus_fingerprint`, `stage1_corpus_fingerprint`, `corpus_mismatch`, `raw_photograph_coverage`), in `metrics.json` under `corpus`, and in `summary.json`'s `split`. |
| **E3** | `readout.stages` (above) plus `layerwise.all_encoders: true`, so the milestone encoders' caches carry stage arrays and the stage-3 readout does not silently fall back to `pooled` for them. |
| **E4** | `summary.json` carries `stage1_transductive: true` and an explicit caveat naming the direction (the setup favours the self-distilled encoder, so a null result is conservative) and the control that would remove it. |
| **E5** | `split_protocol: grouped_cv` in `split_dataset`, plus `merge_out_of_fold()` in the stage-2 trainer: each fold's finished model scores its own held-out half, and the predictions are concatenated **in dataset order** into one out-of-fold set. Every split now reports `classes_present_in_test` and `num_classes`. Opt-in, because no published number used it. |
| **E7** | `dynamics_panels` plots `KL(q‖p)` as the primary panel and the three-way decomposition second, falling back to the raw loss for a run recorded before this existed. `alignment`, `same_image_minus_same_class` and `self_retrieval_top1` are columns in `encoder_comparison.csv`. |
| **E8** | Both caveats added: the CKA-against-`random_init` warning (D12) sits next to Finding 2.1, and the GPU-idle sentence is corrected in place with the derived TFLOP/s figure struck through. |
| **E9** | `nuisance_decodability()` — within-sub-variety photograph identity by nearest-class-mean over stratified folds, with the per-class chance level, reported for every encoder in `tables/nuisance_decodability.csv` and as three columns of `encoder_comparison.csv`. |

### F — optional architectural experiments

| # | What was built |
| --- | --- |
| **F1** | `data.augmentation.same_photo_local_views: n` replaces the trailing *n* local views with local crops of other crops from the same source photograph. The view **count**, `view_ids` and every downstream shape are unchanged. The partner draw is seeded on `(seed, sample index)`, so it is a deterministic function of the sample rather than of which worker built it; a single-crop photograph falls back to augmenting the anchor. The `pickle_batches` format refuses it (no provenance). The trainer logs a warning naming E9 as the gate. |
| **F2** | `experiment.evaluation.prototypes.probe_bottleneck` runs the identical out-of-fold readout over the DINO head's 256-D bottleneck, reported next to the trunk's own readouts, with the caveat attached. |

### Infrastructure

* **`scripts/run_stage1_ablations.py`** — the runner the plan says the repo needs. Trains each arm, evaluates it, and collects one table, with `experiment.training.save_path`, `shared_backbone_path` and `experiment.evaluation.save_path` **all pinned per arm**. Arms are **data, not code**: `conf/stage1_arms/phase0.yaml` … `phase3.yaml`, each carrying the sequence's own rules about what may not be combined.
* **`conf/experiment/eval_frozen_reference.yaml`** (`python main.py eval-frozen`) — `P1-F` as a first-class experiment rather than a wall of CLI overrides.
* **CLI** — `screen-backbones` and `eval-frozen` stages; `--gpus` refused for both, as for `eval-pretrain`.
* **Docs** — `CLAUDE.md`, `README.md`, `SERVER_RUN_GUIDE.md`, `conf/README.md`, `scripts/README.md`, `tests/README.md`, `src/{datasets,losses,models,trainers,utils}/README.md`, `architecture/02` and `architecture/08`, and the corrections in `STAGE1_EVALUATION.md`.

## 2. Intentionally not implemented

| Item | Why |
| --- | --- |
| **B4 — actually cropping the 18 photographs** | Outside this repository: the cropping script that produced `Cropped_Samples` is not here, and the plan itself says to inspect the photographs first because an out-of-focus frame in the SSL corpus is worse than nothing. What *is* implemented is everything that makes the change safe and legible when it happens: the coverage report, the config key, and the corpus fingerprint that keeps results from the old and new corpora distinguishable. |
| **C7 — layer-wise learning-rate decay** | The `lr_base=0.00025` half of C7 is a config override and works today. Layer-wise decay was not added, on the plan's **own** reasoning: *"CKA is stable at `layers.0` (0.976) and `layers.1` (0.960), so the low-level filters did not in fact drift… Layer-wise decay attacks it in the wrong direction (it lowers the LR of *early* layers, which are already stable). The better-targeted version of this idea is C4."* C4 is implemented. Adding a mechanism the specification argues is directionally wrong would be an unused code path on the critical trainer. |
| **C9 — raising the physical batch** | `data.batch_size` and `effective_batch_size` are unchanged at 32. The plan is explicit that C9 should be set **once, together with the trunk decision (B2), or left alone entirely**, and that it must never run as an arm beside C3 because both move the entropy floor `log(K/B_teacher)`. The trunk is not decided until `screen-backbones` runs, so changing it now would pre-empt a decision the plan reserves for Phase 0. `tests/test_configs.py`'s assertion on `data.batch_size == 32` therefore also stands unchanged. |
| **B2/B5 — adopting a new trunk or resolution as the default** | Same reason. `conf/model/backbone/swinv2.yaml` still selects SwinV2-Small. The plan's D6 is explicit that capacity is not the binding constraint and that the IN-1k Base control **decides** B2; that control now exists and has not been run. Changing the default before it runs would be the exact confound the control exists to prevent. |
| **F3 — scale-preserving input (pad-to-square)** | Confidence **low** in the plan, and it "breaks comparability with every published number" while leaving most tokens looking at padding on the 51 % of crops under 64×64. Not built. |
| **F4 — input below 192 px** | Confidence **low**, and the plan calls it "a real experiment" needing the `swinv2_cr_*` family — a different architecture with different weights and only `sw_in1k` tags. `validate_swinv2_name` already admits those names and `data.image_size` is a config key, so it is reachable; nothing was added to encourage it. |
| **D1–D12** | These are the plan's "do **not** do" list. Nothing here implements any of them, and several are now actively guarded: `koleo_scope=all_views` warns, the raw loss is demoted in every figure and log line (D8), `grouped_cv` is opt-in rather than replacing `grouped` (D9), `distributed_sinkhorn` stays off (D10), and the CKA caveat (D12) is written next to the number it qualifies. |

## 3. Deviations that were technically necessary

1. **The reported `loss` and the decomposed `CE` are separate keys.** With KoLeo or an auxiliary head enabled, `loss` is the whole objective and does **not** equal `H + KL`. The Eq. 1 term is logged separately as `dino_cross_entropy`, and that is what decomposes exactly. The log line shows both (`loss=… | CE=… = KL … + H …`).

2. **`teacher_entropy` keeps its historical definition.** The plan's `loss − teacher_entropy` is exact only when every teacher view takes part in the same number of cross-view pairs — true for the shipped 2×6 geometry, not in general. The implementation adds `teacher_entropy_cross_view`, weighted by each teacher view's share of the pairs, so the identity is exact for any mask; `teacher_entropy` remains the plain mean over teacher rows, so a number logged before this change and one logged after mean the same thing.

3. **The KL decomposition is not gated on `metrics_enabled`.** The plan describes it as "one subtraction, no synchronisation", which is right — but the collapse diagnostics around it *are* gated, and an epoch mean computed only on logging steps would not be an epoch mean. It is computed every micro-batch, as device tensors, and `tests/test_throughput.py` was updated to pin the new (deliberate) split between gated and ungated diagnostics.

4. **`measure_gpu_busy` drains its CUDA events in batches.** Synchronising on every micro-batch — the obvious reading — would stall the loop once per micro-batch and change the very throughput it is trying to measure. Events are queued and drained at logging steps and at the epoch boundary, so at most `log_every_steps` pairs are outstanding.

5. **`grouped_cv` ignores `experiment.training.max_batches` for the out-of-fold pass.** Full coverage of the fold *is* the protocol; a truncated pass leaves the assembly with fewer predictions than indices, and `merge_out_of_fold` raises. `max_batches` is a smoke knob, and honouring it here would turn the smoke run's failure mode from "fast" into "incoherent".

6. **The arm-suite runner forwards only `data.*` overrides to the evaluation.** `experiment.training.*` has no meaning in an evaluation config and Hydra's struct mode rejects it, so forwarding the whole `common` block would make every arm's evaluation fail. `data.*` is exactly the set the alignment measurement must reproduce.

7. **`P1-F` is a Hydra experiment, not CLI overrides.** Expressing "remove every encoder row that reads a stage-1 artifact" as command-line overrides needs a nested list literal and three `enabled=false` flags; `conf/experiment/eval_frozen_reference.yaml` is the same thing as a reviewable file.

8. **Multiple GPUs are not sharded across stage-1 arms.** `run_ablations.py` shards stage-2 variants one per device because each is small. A stage-1 arm saturates one device, so the runner prints a note and runs arms sequentially — the plan's own G4 says to use both devices *within* an arm.

## 4. Verification performed

| Check | Result |
| --- | --- |
| `pytest tests/` | **603 passed** (from 518). New: `tests/test_stage1_changes.py` (78 tests) and two real 2-rank gloo tests in `tests/test_distributed.py` |
| `ruff check --select E9,F` on every changed/new file | clean (6 pre-existing unused imports in `pretrain_eval.py` left alone) |
| `python -m compileall src scripts tests main.py` | clean |
| All 15 `conf/experiment/*.yaml` compose and resolve | pass |
| `main.py smoke` (both stages) | pass |
| `scripts/dry_run.py` | pass |
| Stage-1 smoke with `aux_stage=2`, `koleo_space=backbone`, `koleo_scope=all_views`, `same_photo_local_views=2`, `match_view_lowpass=true` together | pass; `loss = KL + H` verified numerically |
| Stage-1 evaluation end to end, and `eval-frozen` end to end | pass; `readout_stages.csv` and `nuisance_decodability.csv` produced |
| Stage-2 smoke with `grouped_cv` + `feature_stage=stage3_pooled_2x2` | pass; out-of-fold over all 324 synthetic crops, all 27 classes |
| A stage-1 checkpoint loads at every `feature_stage`, strict, zero missing keys | pass |
| An aux-head run's published trunk loads as a plain trunk, strict | pass |
| `resolve_accumulation` / `resolve_learning_rate` / `resolve_num_workers` at world sizes 1 and 2 | pass |
| `run_stage1_ablations.py --dry-run` for every manifest | pass; per-arm paths distinct |
| `git diff` reviewed for unrelated changes | every deletion is an intentional replacement |

**One check could not be completed on this machine.** A full 2-rank stage-1 job
under `torch.distributed.run` hangs at rendezvous — and so does a three-line
`init_process_group` + `all_reduce` probe, so the hang is environmental (macOS
c10d store), not a property of these changes. Real multi-process gloo *does* work
here: `tests/test_distributed.py` spawns genuine 2-rank jobs through `mp.spawn`
with a file-based init and all 12 pass, including the two added for the stage-1
changes. **Run `python main.py pretrain --gpus 2` once on the target server
before launching Phase 1.**

## 5. Launching the phases

```bash
export SEED_DATA_ROOT=/path/to/Hierarchical_SeedData/Cropped_Samples
export SEED_RAW_DATA_ROOT=/path/to/Hierarchical_SeedData/RAW_Samples   # optional, enables the coverage audit
export SEED_OUTPUT_DIR=/path/to/outputs

# ---- Phase 0: no training ------------------------------------------------
python scripts/report_raw_photographs.py          # 0.6 — inspect before cropping
python main.py screen-backbones                   # 0.5 — pick the trunk and resolution
python main.py eval-pretrain                      # 0.4 — confirm the stage-3 readout
python main.py eval-frozen                        # the reference every arm must beat
# or both screens at once:
python scripts/run_stage1_ablations.py --arms conf/stage1_arms/phase0.yaml

# 0.2 — the throughput gate. `epoch/loop_blocked_fraction` must fall below ~0.3.
python main.py pretrain data.num_workers=16 experiment.training.epochs=2

# ---- Phase 1: five arms + the frozen reference ---------------------------
python scripts/run_stage1_ablations.py --arms conf/stage1_arms/phase1.yaml \
    --experiment <the trunk Phase 0 chose>        # e.g. pretrain_swinv2_tiny_dino
# preview first:
python scripts/run_stage1_ablations.py --arms conf/stage1_arms/phase1.yaml --dry-run

# ---- Phase 2: resolve the survivors --------------------------------------
# Paste the Phase-1 winner's overrides into conf/stage1_arms/phase2.yaml `common`,
# then:
python scripts/run_stage1_ablations.py --arms conf/stage1_arms/phase2.yaml \
    --experiment <the trunk Phase 0 chose>

# ---- Phase 3: confirm, with error bars -----------------------------------
python scripts/run_stage1_ablations.py --arms conf/stage1_arms/phase3.yaml \
    --experiment <the trunk Phase 0 chose> --seeds 42 43 44

# ---- Stage 2 against a chosen arm's encoder ------------------------------
SEED_PRETRAIN_BACKBONE=$SEED_OUTPUT_DIR/stage1_arms/P1-A/encoder.pth \
    python main.py finetune \
        model.backbone.feature_stage=stage3_pooled_2x2 \
        experiment.training.split_protocol=grouped_cv
```

Results land in `$SEED_OUTPUT_DIR/stage1_arms/stage1_arm_results.csv`, headlined
on `oof_probe_sub_accuracy_testable_classes` with the fold SD, and carrying
`final_teacher_student_kl` (the learnable half of the objective) and
`nuisance_photo_above_chance` (the falsifier — an arm that *raises* it may have
won the probe by re-learning the photograph confound).

**Read `P1-0` against `P1-F` first.** That is the comparison this whole sequence
exists to make, and if it does not clear a fold SD the answer is that stage 1 as
an objective is not earning its compute.
