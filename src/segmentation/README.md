# `src/segmentation/` — stage 0: photographs → corpus

| File | Contents |
| --- | --- |
| `illumination.py` | The photometric model of one scene: illumination field, paper (support) region |
| `detect.py` | Foreground scoring, binarisation, connected components, watershed instance separation |
| `instances.py` | `SeedInstance`, its descriptors and verdict; the square crop policy; distractor suppression |
| `pipeline.py` | One photograph in, adjudicated detections out — and the scene gate |
| `extract.py` | Hydra entry point: writes the refined corpus, the manifest and the figures |
| `audit.py` | Hydra entry point: validates a corpus against the recovered legacy boxes |
| `benchmark.py` | Does the refined corpus read out better? Two corpora, one frozen encoder, no training |
| `visualize.py` | Overlays, rejection galleries, before/after panels |

```bash
python main.py extract-seeds      # RAW_Samples -> Refined_Samples + manifest + overlays
python main.py validate-seeds     # recall, duplicates, coverage, rejection census
python main.py benchmark-corpus   # is it actually better? (no training)
```

Nothing here imports from `models`, `losses` or `trainers`, and nothing there
imports from here. The only thing that crosses is a directory of PNGs in the
layout `HierarchicalSeedDataset` already walks.

## What the data is

99 iPhone-11 photographs, ~8 MP, of seeds scattered on a sheet of paper: 27
sub-varieties under 4 seed types, one variety per photograph. Seeds are 30–150 px
across. Three of the 99 are not trays at all — a near-empty sheet, a paper packet
standing on a wooden table, and a labelled ziplock bag.

The corpus this replaces, `Cropped_Samples`, is 9,357 hand-curated crops from 81
of those photographs. They are **byte-identical sub-images** of the raw files,
which is the single fact that makes this package auditable rather than
self-reported: their exact bounding boxes are recoverable by template matching,
so "did we miss seeds?" is a measurement.

## The method, and why each piece is there

**Global thresholding does not work, and that is measured.** OpenCV's Otsu at
quarter resolution returns a "foreground" fraction spanning **0.9 %–64.7 %**
across the 99 photographs: on a frame whose seeds cover a few percent of the
sheet, the between-class variance is maximised by splitting the *paper's own
illumination gradient*. Three ordinary, well-exposed trays come back above 54 %.

So the scene is modelled instead:

1. **Illumination field** — a morphological closing at 1/16 resolution with an
   element spanning ~500 native px, which is larger than the widest seed
   (149 px) and so interpolates *through* every seed. The paper is a median
   4.3 % brighter at the centre than at the border (range −13.3 % to +9.6 %), so
   a single number cannot describe one frame.
2. **Support region** — the largest bright, low-chroma component with its holes
   filled, then eroded by ~1.5 seed diameters. The erosion is load-bearing: the
   paper's edge and its cast shadow otherwise survive as one component 34× the
   median seed area on `PearlMillet/IMG_0510` and 102× on `Poosa33/IMG_0665`.
3. **A two-channel foreground score**, the pointwise maximum of two robust
   z-scores against the *paper's own* MAD — relative darkness against the field,
   and CIELAB (a, b) distance from the paper's colour. Neither alone covers the
   corpus:

   | photograph | relative darkness (fg / bg) | chroma (fg / bg) |
   | --- | --- | --- |
   | Amaranthus IMG_0653 | 0.39 / 0.99 | 4.5 / 1.0 |
   | KodoMillet IMG_0492 | 0.34 / 0.98 | 7.1 / 1.0 |
   | Chinnar IMG_0689 | 0.61 / 0.99 | 8.1 / 1.0 |
   | **MilaguSamba IMG_0179** | **0.73 / 0.98** | **28.4 / 1.4** |

   `MilaguSamba/IMG_0179` decides the design: its straw-coloured grains sit at
   73 % of the paper's brightness against the paper's own 5th percentile of
   89 %, so a luminance-only detector must choose between missing grains and
   eating the sheet. In chroma the same grains are 20× further from the paper
   than the paper's own spread. Near-black amaranthus is the mirror case.

4. **Instance separation**, only where warranted. A connected component is not a
   seed: 2–8 components per photograph exceed 1.6× the frame's median seed area.
   A component is split only when **all three** of the following hold, because
   no one of them is sufficient:

   | signal | fails alone because |
   | --- | --- |
   | area > 1.5 median seeds | 12 components on `KodoMillet/IMG_0492` measure 1.5–1.85 and every one is a single seed |
   | solidity < 0.93 (a waist) | a merged pair measures 0.77–0.88, a large single 0.92–0.98 — but PearlMillet's hilum notch also measures 0.79 |
   | ≥ 2 distance-transform peaks | **19 of 66** unambiguously single grains on `Chinnar/IMG_0689` carry two or more |

   and the result is then **verified**: a fragment outside 0.45–1.80 median seeds
   discards the whole split, and the component is recorded as an unresolved
   cluster rather than emitted as halves nobody checked.

5. **A scene gate.** The three non-tray photographs are identified by two
   measurements a genuine tray passes with a wide margin — support fraction
   (0.76–1.00 for trays, 0.58 / 0.63 / 0.65 for the three) and plausible seed
   count (31–390 against 2). Not by filename.

Everything the gates reject keeps its descriptors, its reason and a row in
`manifest.csv`, and appears in the overlay and the rejection gallery. The
`REJECTION_REASONS` tuple is the closed set the audit tabulates.

## The crop policy, and the two things it refuses to do

**It does not squash.** The legacy crops are tight, non-square boxes — 3.4 %
square — which the stage-2 transform resizes with an explicit `(H, W)` pair, i.e.
it stretches each crop to a square by a factor that depends on the seed's
*orientation in the photograph*. A rice grain lying flat has a 3:1 box and is
compressed 3× along its length; the same grain at 45° has a square box and is not
compressed at all. That turns a rigid rotation of the object into a shape change,
on a task whose classes differ by shape — and it is exactly what the dihedral
augmentation assumes is *not* happening. The refined crops are square windows cut
from the photograph, so no such factor exists.

**It does not mask.** The seed is emitted with the paper around it. Zeroing the
background would delete the boundary contrast that says where the seed ends,
which on a 40 px amaranthus seed is a large share of the file's information. What
*is* removed is other seeds — see below.

**It does not resample.** Every emitted pixel is a source pixel. The window is
cut, never resized, so the only resize in the whole pipeline is the one the
augmentation applies: explicit, configurable, and in one place.

The margin is measured, not chosen. Over 1,107 matched pairs the legacy crops
carry a median 5–6 px of padding (~14 % of the tight side on the small classes),
and `margin = 0.12` reproduces that ring while guaranteeing the whole boundary is
inside the frame. Its cost is measured too — neighbour intrusion into the window
rises 1.6 % → 2.9 % between margin 0 and 0.12, and 6.5 % at 0.30.

**Distractor suppression** handles that 2.9 %. A neighbouring seed inside the
window is inpainted out with Telea from the paper around it, and the target's own
pixels are never touched. Telea is right here for a reason specific to this
imagery: the surround is a smooth, textureless sheet, so the fill reproduces its
gradient and grain rather than pasting a flat patch that would itself read as an
object. Which crops were edited, by how much, and which labels were removed all
go into the manifest.

## What the audit measures

`python main.py validate-seeds` writes `outputs/segmentation/audit.json` plus
`legacy_comparison.csv` and two figures. On the shipped extraction:

| | |
| --- | --- |
| legacy crops located, byte-exact | 9,050 of 9,357 |
| recall of them by the refined detector | **99.71 %** (9,024) |
| legacy-only, i.e. possibly missed | **26** — individually listed and pictured |
| refined-only, i.e. newly found in the same photographs | **1,439** |
| duplicate pairs (same seed, two files) | **0** refined, **0** legacy |

The 307 crops that are *not* byte-exact all belong to four photographs —
`Chithrakar/IMG_0161`, `IMG_0162`, `Kullakar/IMG_0711`, `IMG_0712` — whose raw
file was replaced by a differently oriented version after the crops were cut
(mean absolute pixel difference 15–22 at the recovered location, against exactly
0 everywhere else). Their recovered boxes describe nothing, so they are named and
excluded from the denominator rather than scored.

Of the 26 legacy-only cases, roughly a fifth are legacy crops of the **wooden
table or a paper edge** rather than of a seed; the rest are seeds with an
attached awn or husk, and seeds clipped by the frame border. `figures/legacy_only_seeds.png`
shows all of them at native resolution.

## Is the new corpus actually better?

`python main.py benchmark-corpus` — two corpora, the same frozen encoders, the
same protocols, no training. The size- and source-matched control is on by
default, because "more crops" is not "better crops".

Frozen ImageNet-1k SwinV2-Tiny, 27-way linear probe (`C` selected on a validation
split, never on test):

| protocol | legacy 9,357 / 81 | refined 13,492 / 96 | refined, matched to 9,347 / 81 |
| --- | --- | --- | --- |
| crop-level probe | 0.8007 | **0.8599** | 0.8449 |
| crop-level macro-F1 | 0.8004 | 0.8416 | 0.8424 |
| crop-level k-NN | 0.7099 | 0.7629 | 0.7235 |
| photograph-disjoint (out-of-fold) | 0.6194 | **0.6840** | 0.6371 |

The matched column is the one that isolates the crops: **+4.42 pp** crop-level
and **+1.77 pp** photograph-disjoint from the crop redefinition alone, with the
rest coming from the 15 additional photographs — which is the axis this dataset
is actually short of.

Both shortcut floors move the *other* way, which is the result worth having:

| encoder (crop-level) | legacy | refined | gap to `imagenet_init` |
| --- | --- | --- | --- |
| `handcrafted` (10 scalars) | 0.5075 | 0.4983 | +29.3 pp → **+36.2 pp** |
| `random_init` | 0.5903 | 0.5169 | +21.0 pp → **+34.3 pp** |

Ten trivial statistics score *lower* on the refined corpus, largely because
`log(aspect ratio)` — a real shape cue that used to sit in the file's dimensions
— is now constant at 0. The cue did not disappear; it moved into the pixels,
undistorted, where an encoder has to look at the seed to use it. That is the
trade the square crop makes, and it is the direction a fine-grained task wants.

## Adding a rejection reason

1. Add it to `REJECTION_REASONS` in `instances.py` — `SeedInstance.reject`
   refuses anything not declared, so a reason invented at a call site cannot
   become a category the audit silently omits.
2. Apply it in `segment_photograph`, in the right pass: size gates run first,
   then shape/photometry against the photograph's own population.
3. Give it a colour in `VERDICT_COLOURS` so it is visible in the overlay.

Nothing else needs changing: the manifest columns, the per-photograph summary,
the rejection census and the gallery are all derived from the reason list.
