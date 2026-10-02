# CALLISTO Trainer

A desktop tool for building a **region-level** labelled dataset of solar radio
bursts from e-CALLISTO FITS files, and for training, validating and testing
models on it — all from the raw dynamic-spectrum array, never from rendered
images.

Import FITS batches, review each preprocessed spectrum, decide burst / no-burst,
draw a box around each burst and give it a type -- Type II, Type III, **Type
IIIG** (a group of Type III bursts), **Type IV**, Other. Interference is never
drawn: the program finds it itself, outside your burst boxes and in no-burst
files, and trains the model to reject it as **RFI**. Work commits to SQLite as
you go, so the app can be closed or killed at any moment and resumes exactly
where it stopped.

Built on the pipeline in `H:\Burst Identifier`, whose preprocessing is vendored
here byte-for-byte (see [Relationship to Burst Identifier](#relationship-to-burst-identifier)).

---

## Quick start

```bash
python -m venv .venv
```

```bash
.venv\Scripts\pip install -r requirements.txt
```

For GPU training, install the CUDA-matched torch build instead of the pinned one
(this machine uses cu132 on an RTX 5060):

```bash
.venv\Scripts\pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu132
```

Run it:

```bash
.venv\Scripts\python -m callisto_trainer.app
```

---

## The workflow

**Import** → **Label** → **Dataset** → **Train** → **Evaluate** → **Predict**

1. **Import** — point at folders of `.fit.gz`. Files are referenced in place and
   never copied, so importing from a large archive costs no extra disk. Re-importing
   is safe: duplicates are caught by path *and* by content hash.
2. **Label** — review each spectrum, mark burst / no burst, drag boxes around
   individual bursts and type them. Several bursts per file, each typed separately.
   Only bursts are boxed; leave interference alone.
3. **Dataset** — freeze the current labels into an immutable, timestamped snapshot:
   tensors, a manifest with the train/val/test split, and a ready-to-run config.
4. **Train** — pick a snapshot, adjust the run, watch loss and score curves live.
5. **Evaluate** — run a checkpoint on the held-out split; double-click any
   misclassified sample to jump back to it in the Label tab. Export the model
   from here as a portable bundle.
6. **Predict** — run trained models over new, unlabelled files and export the
   results as CSV or JSON.

### Keyboard

| Key | Action |
|---|---|
| `A` / `D` (or arrows) | previous / next file |
| `F` | next unreviewed file |
| `B` / `N` / `U` | Burst / No burst / Unsure |
| `1` / `2` / `3` / `4` / `5` | set selected box to Type II / Type III / Type IIIG / Type IV / Other |
| `P` | suggest candidate regions (the ones a trained unified model calls bursts) |
| `V` | cycle preprocessed / raw / quiet-background view |
| `Del` | delete selected box |
| `R` | reset zoom |
| drag | draw a burst box |
| shift-drag, middle-drag | pan |
| scroll | zoom |

### The labels

| Key | Type | Mark it when |
|---|---|---|
| `1` | Type II | a slowly drifting band (shock), often with harmonic lanes |
| `2` | Type III | a single fast-drifting burst |
| `3` | **Type IIIG** | a **group of 3 or more** Type III bursts in quick succession |
| `4` | **Type IV** | a broadband continuum lasting minutes to hours, stationary or slowly drifting, often with fine structure |
| `5` | Other | any other solar burst (Type I, V, U, J, spikes...) |

There is no RFI key. Everything the region finder proposes **outside your burst
boxes**, and everything in a file marked **No burst**, is already known not to be
a burst; the exporter measures each of those regions and names the ones with an
interference signature RFI itself (see *Interference is found, not drawn* below).

Rules the Label tab enforces, so the verdict and the boxes never disagree:

- A burst box implies the file has a burst, as before -- also on a file that was
  marked *No burst*.
- If the only burst box on a file is deleted, a verdict that drawing it had set
  is withdrawn. Pressing `B` / `N` / `U` is always final.
- Marking a file *No burst* asks before deleting its burst boxes.

**Long Type IV continua.** The preprocessed view subtracts each channel's median
over the file, so a continuum lasting more than about half the 15-minute file
looks flattened there (the quiet part turns dark instead). Switch the canvas to
**Quiet background** (`V`) to see its true extent before boxing it: each
channel's background is taken from its quietest tenth instead, on the same dB
window. The model is given the same view (see *What the model sees*).

The panel counts the **separate bursts** inside the selected box (runs of bright
samples per channel, drift-independent, cadence-aware) and suggests Type IIIG
for a Type III box holding three or more. The count is advisory, never applied.

**Rare classes fold until they can be learned.** Type IIIG trains as Type III
and Type IV as Other (what it was labelled before it had a class) until each has
20 drawn boxes, and automatic RFI trains as No_Burst until it has 20 samples in
every split; the Dataset tab says so. A class learned from a handful of boxes is
memorised, and it would also block training by leaving a split without it. A
fresh dataset therefore trains from day one.

### Seeing the underlying data

The canvas necessarily shows an *interpretation* of the file. The **Raw data**
tab and the `V` key expose what is beneath it:

- **View** (`V` cycles) — the normalized view the model receives; the array
  exactly as stored in the FITS (uncalibrated receiver digits, no background
  removed, the instrument's frequency-dependent gain bands plainly visible); and
  the **quiet background**, the model's third view, where a continuum lasting
  most of the file stays bright. Zoom and boxes never move when switching.
- **FITS header panel** — the full primary header, the extension list, and an
  explicit note on whether the frequency axis came from the `AXES` table or the
  unreliable header fallback.
- **Pixel inspector** — the same sample at all three stages: raw digit, dB above
  background, and the 0–1 number fed to the network. It flags samples that are
  saturated at the edge of the training window, and samples that are `NaN` in the
  file (preprocessing substitutes the finite median, so the model sees a
  fabricated value there — worth knowing).

### Resetting

Under the **Reset** menu:

| Action | Effect |
|---|---|
| Reset view and layout | zoom, contrast, colormap, filters, panel sizes. Labels untouched. |
| Clear this file's labels | drops the verdict and every box for the open file; it returns to unreviewed |
| Clear display cache | forces every file to be re-read from the original FITS |
| Reset entire dataset | deletes all files, boxes and verdicts. Requires typing `RESET`, and writes a timestamped database backup first. Your FITS files are never touched. |

### Managing trained models

`Reset → Manage trained models...` lists every training run with its date, best
score, checkpoint count and size, alongside the dataset snapshot it came from.

Runs are expensive: each checkpoint is ~134 MB (45 MB of weights, **89 MB of
AdamW optimizer state**) and a run keeps `best.pt`, `last.pt` and up to three
per-epoch copies — about 640 MB each. Six runs reached 3.6 GB here, with 2.6 GB of
snapshots behind them.

So there are three levels, not just delete:

| Action | Frees | Costs you |
|---|---|---|
| **Free space** | ~384 MB/run | nothing usable — the epoch copies exist only to roll a run back to an earlier peak; `best.pt` and `last.pt` survive |
| **Archive** | ~171 MB/run more | the ability to *resume* that run. It still evaluates, exports and predicts identically |
| **Delete model** | ~640 MB/run | the model. The snapshot is kept, so it can be retrained |
| **Delete model + dataset** | ~1 GB/run | both. Your annotations are untouched; a snapshot can be re-exported any time |

Prune + archive together take a run from **640 MB to 85 MB** — a 7.5× reduction
with a model that still predicts, which `tests/test_model_registry.py` asserts by
loading an archived checkpoint and running it.

Deletion is confined to the configured `outputs/` and `datasets/` directories; a
path that escapes them is refused rather than trusted.

---

## How a box becomes a training sample

This is the core of the design, and the one rule that must not be broken.

The background step subtracts the **per-frequency median along the time axis**.
If a box were cropped from the raw array and preprocessed afterwards, that median
would be computed over only the few seconds inside the box — and a burst filling
the box would be subtracted into its own background, erasing exactly the signal
being labelled.

So the order is fixed:

```
read → clean → background-subtract → normalize      (whole file)
                                          │
                                          └─→ slice the box
                                                   │
                                                   └─→ resize to 224×224
```

Every step except the slice calls the vendored pipeline functions unchanged. The
direct consequence — asserted by `tests/test_crop_equivalence.py` — is that a
full-extent crop is **bit-identical** to the original `preprocess_array()`.

Boxes are stored twice: as native pixel indices (exact, always available, used
for cropping) and as physical coordinates in MHz and seconds (portable, and what
a future detection model would need). Nothing about the annotation is lossy, so
the same labels can later feed a true detector without re-annotating.

The crop is **exactly the box that was drawn** — the tensor previewed beside the
canvas is the tensor training receives, pixel for pixel, so a sample can be
verified by looking at it. The `crops` config section can pad a box with
surrounding context (`context_margin`) or grow a tiny one to a floor size
(`min_rows` / `min_cols`), but both default to off; raising either makes every
tensor cover more than was labelled, so only do it if the whole corpus is
re-exported with the same values.

---

## The unified model (default)

One softmax applied to a *region*. It answers everything at once: whether a
region is a burst, which type it is (Type II / III / IIIG / IV / Other), and,
applied across a file's candidate regions, where the bursts are. A region is a
burst when its **burst evidence** -- one minus the probability that it is not a
burst -- reaches the model's **calibrated threshold**; a file is a burst if any
region is. Which *type* a burst is called is corrected for how often each type
really occurs (see *Type frequencies* below).

**RFI and No_Burst are one outcome in everything you see** -- predictions,
evaluation reports, confusion matrices, class counts: *not a burst*. Inside,
the model is still trained with RFI as its own rejection class, because that
split cut false alarms from 6 to 1 of 207 held-out quiet files (see *Measured,
per file*). **RFI is detected separately**, among the regions that are not
bursts: a region is reported as RFI when its measured features carry an
interference signature (the table below) or the model's own RFI output outweighs
its background output. Regions on the same channels are grouped into one
**interference source** -- the periodic calibration block in a station's lowest
channels comes out of the region finder as about 14 segments in every file, and
is reported once. RFI never decides a verdict:

- a file with a burst is **Burst**, whatever interference it also holds, and
  lists that interference beside the bursts (`Type III  ·  RFI x1`);
- a file with only interference is **No_Burst**, and lists it (`RFI x2`);
- a burst region is never turned into RFI, whatever it overlaps.

### Interference is found, not drawn

RFI is never boxed. Every region the finder proposes in a file marked *No burst*,
and every region of a burst file outside the drawn burst boxes, is a rejection
already -- the verdict and the boxes say so. The exporter measures each one's
region features and names it **RFI** when it has an interference signature
(`core/rfi_labels.py`), otherwise leaves it **No_Burst**:

| Signature | Rule (region features) |
|---|---|
| impulse | lights > 60% of the band, onset spread ≈ 0, ≤ 1 s thick |
| sweep | a sparse track over > 40% of the band, ≤ 0.5 s thick, never lit all at once |
| periodic | comb periodicity > 0.5 |
| gain step | a flat plateau over > 60% of the band |
| carrier | its channels stay bright outside the region > 40% of the time, or a ≤ 22-channel line lasting > 40% of the file |
| flagged channels | the station's own `RFI_FREQ` table covers the region |

The exporter uses the same rules to name the RFI the model trains on; the
Predict tab uses them to report it. The thresholds were set on 22,693 regions
from 555 archived files. They name
**0% of the regions on real Type III boxes, 0.9% on Type II and 0.3% on Other**,
against 85-100% of each synthetic interference pattern and about a quarter of
all background regions. Looked at by eye, the background regions they name are
interference -- above all the periodic calibration blocks in the lowest channels
of many stations, then keyed carriers and repeating sweeps; the ones they leave
alone are mostly weaker carrier fragments and noisy bands. The rules are strict
on purpose: naming a burst RFI would teach the model to reject bursts, while
leaving interference as No_Burst costs nothing, because both are rejections and
the burst decision adds them together. Synthetic interference (below) still adds
the shapes that are rare in real files.

### Why the previous version raised so many false alarms

Its test report said **98.8% of background crops rejected**, and in use it
flagged far too many quiet files. Both were true, and the gap had five causes:

1. **Crops are not files.** A file is examined region by region -- up to a dozen
   candidates -- and called a burst if *any* is. A 1.2% per-region error over
   twelve regions is roughly a 13% per-file false-alarm rate. Nothing measured it.
2. **No operating point.** The decision was the argmax, with class weights that
   *cheapened* background: the more background samples, the less each counted.
3. **Noisy positives.** Any finder region ≥60% inside a drawn box took its type.
   Type II and Other boxes averaged ~1,050 × 105 px, so interference inside them
   was trained as burst.
4. **Thin negatives.** About one background crop per burst crop, capped at 4 per
   burst file, and never the model's own mistakes.
5. **Blind to what makes RFI RFI.** Every region is resized to 224×224, which
   destroys the evidence: that a carrier continues across the whole file, that an
   impulse starts at every frequency in the same sample, that a sweep is one
   sample thin, that interference repeats. Train macro-F1 reached 1.00 against
   0.80 on validation: it memorised rather than learned what to reject.

Each is addressed below, and the fix is measured at file level, not crop level.

### What the model sees

Every region reaches the model three ways, built by one encoder
(`core/region_inputs.py`) for export, calibration, prediction and suggestions:

| Input | What it shows |
|---|---|
| **crop** | the region exactly, as before -- bit-identical to the labelling preview |
| **context view** | the full band over several times the region's duration, max-pooled so a one-sample spike survives. Previewed beside the crop |
| **quiet-background context** | the same strip, from the file normalized with each channel's **10th percentile** as its background instead of its median. Previewed third |
| **28 region features** | the 8 drift/extent physics features below, plus 20 interference features |

The views share one backbone (they are not pixel-aligned, so they are embedded
separately rather than stacked as channels).

The third view exists for **Type IV**. The median background makes a continuum
that lasts more than about half the file its own background: on the archive's
long "Other" boxes (over 80% of the file) only 0.7% of the pixels cleared the
region finder's level, against 3-9% for every other kind of box. Taking the
background from the quiet part keeps such a continuum bright while leaving short
bursts as they were. It cannot help a continuum that fills the whole file, which
has no quiet part in any per-file background, and the region finder still runs
on the median view -- the third view changes what the model sees of a candidate,
not which candidates there are. The Dataset tab option *Add the quiet-background
view* switches it off; snapshots are half again as large with it.

Measured as an A/B on the archive: two models trained identically (12 epochs,
automatic RFI, type correction), one with the third view, judged on the same
held-out files. (Mid-benchmark 413 of the labelled files -- the `Burst List`
folder, most of the Type II examples -- disappeared from disk, so both arms ran
on the 1,690 that remained: 192 quiet and 60 burst test files, and 13 long boxes,
over half the file wide, in the validation and test files.) At an equal number
of false alarms:

| test false alarms | 0 | 1 | 3 | 4 | 9 |
|---|---|---|---|---|---|
| burst files found (60), two views | 7 | 39 | 43 | 43 | 46 |
| burst files found, **three views** | **31** | 39 | **45** | **45** | **49** |
| long boxes found (13), two views | 3 | 8 | 9 | 9 | 11 |
| long boxes found, **three views** | **5** | **9** | **11** | **11** | **13** |

The third view ranks bursts better, above all the long boxes it exists for. At
their own calibrated thresholds the two models land at different points of that
curve: the two-view one at 4 false alarms / 43 files / 9 of 13 long boxes, the
three-view one at 11 / 49 / 13 of 13, because its recall kept rising inside the
5% validation budget and the calibration spends the budget on recall. If you
would rather hold false alarms near 2%, set `calibration.max_false_alarm_rate`
to 0.02 in the Train tab's config: at 4 false alarms the three-view model still
finds 2 more burst files and 2 more long boxes. The samples are small -- 13 long
boxes, and retraining moves counts by about 2 -- so this is a consistent
direction, not a precise size. The interference features
(`core/region_features.py`) each target a false-positive pattern:

| Pattern | Feature |
|---|---|
| narrowband carrier | `outside_persistence` -- are these channels bright elsewhere in the file; `log_thickness_freq` |
| broadband impulse | `band_fraction`, `simultaneity` (whole band lit in the same sample), `log_onset_spread` ≈ 0 |
| sweep / chirp | `log_thickness_time` -- one or two samples thin |
| periodic signal | `periodicity` -- an autocorrelation *comb* test; white noise scores < 0.15 |
| gain step, switching | `peakiness` ≈ 0 with `band_fraction`, `fill_fraction` ≈ 1 |
| noisy station | `log_snr`, `fill_fraction`, `file_bright_fraction`, `file_noise` |
| flagged channels | `rfi_flag_fraction` -- the station's own `RFI_FREQ` table, previously unused |
| Type IIIG | `burst_count` |

### Station and date: a capped correction

The model also takes the file's **station** and **observation month and year**.
Both carry real signal -- each station has its own receiver and its own
interference, and solar activity and the ionosphere change with the season and
the cycle -- but in labelled data they are also a shortcut. How often a
station's files hold a burst mostly reflects which of its files were picked for
labelling (USA-BOSTON: 4 bursts in 287 labelled files; ALASKA-COHOE: 295 in
671), and a held-out split drawn from the same labels cannot tell the two apart.
So station and date are not fused freely with the image; they enter as a
**bounded correction** on top of what the image and region features say
(`models/physics_model.py`, `StationDateCorrection`):

* **Capped.** Each class score moves by at most `cap / 2` through a `tanh`, so
  no log-odds -- burst vs not a burst, or one type vs another -- moves by more
  than `cap`. At the default `cap = 1`, a region the image puts at 50% can end
  up between 27% and 73%; one at 95% no lower than 87%.
* **Interactions, not just priors.** The correction also reads the image
  representation, so it can learn "this narrowband line, at this station, is its
  known carrier" -- within the same cap.
* **The image path stands alone.** The correction starts at zero, and during
  training station and date are hidden at random -- together (25% of samples),
  so the image path is trained on its own, and separately (15% each), so either
  can be missing later. With both unknown the correction is exactly zero.
* **Encoding** (`metadata_features.StationDateEncoder`): the station's index if
  it has at least 10 training files (others, and stations never seen, share an
  "unknown" slot); the month as a point on the yearly cycle; the year as a
  number clamped to the trained span, so a file from after the last trained
  month is treated like that month rather than extrapolated to. The station list
  and span are fitted on the training split when training starts and saved in
  the checkpoint.

The Train tab's **Station + date** setting chooses the cap (off, 0.5, 1 or 2).
Every file-level report (`val_file_metrics.json`, `test_file_metrics.json`) then
has a `station_date_effect` section: each file is judged twice, as scored and
with station and date hidden, and every file whose verdict differs is listed --
false alarms added or removed, bursts found or lost because of them.

### Backbones

`resnet18` (default), `resnet34`, `resnet50` and `convnext_tiny` (plus the older
`efficientnet_b0` and `mobilenet_v3_small`), all ImageNet-pretrained. On an 8 GB
GPU, ResNet50 needs a batch of at most 32 regions and ConvNeXt-Tiny at most 24;
the Train tab lowers it when you pick one. Above that, Windows does not fail with
out-of-memory -- it spills into system memory and training runs several times
slower (ConvNeXt-Tiny at 32 crept over after nine epochs and went from 3 to 13
minutes an epoch). The same happens if another program -- or this app's own
Predict or Label tab holding a model -- is using GPU memory while a large
backbone trains; the sign is the GPU at 100% while drawing about half its power.

`performance.channels_last` is now **off** by default. Measured on an RTX 5060
(torch 2.12, CUDA 13), it made ResNet training 5-8x slower (ResNet18 243 vs
1,402 images/s, ResNet50 58 vs 473) and ConvNeXt no faster.

### What station/date and the deeper backbones are worth (measured)

One snapshot of all 3,512 labelled files (25,054 regions), every model trained 20
epochs and calibrated to the 5% budget on the validation files, then judged on
the **test files** -- 335 quiet, 189 with bursts -- exactly as Predict runs.
"Found at 2% / 5% FA" counts burst files found at the threshold where 2% / 5% of
the quiet test files are flagged, the same rule for every model, so it compares
ranking without depending on where calibration happened to land. Each station/date
model is also scored with station and date hidden.

| model | false alarms (of 335) | bursts found (of 189) | found at 2% FA | found at 5% FA | train |
|---|---|---|---|---|---|
| ResNet18, no station/date, seed 42 | 19 | 141 | 115 | 137 | 18 min |
| ResNet18, no station/date, seed 7 | 20 | 145 | 114 | 144 | 18 min |
| ResNet18 + station/date, seed 42 | 19 | 142 | 117 | 137 | 18 min |
| &nbsp;&nbsp;same model, station/date hidden | 24 | 141 | 115 | 138 | |
| ResNet18 + station/date, seed 7 | 15 | 142 | 130 | 143 | 18 min |
| &nbsp;&nbsp;same model, station/date hidden | 8 | 135 | 132 | 142 | |
| ResNet34 + station/date | 17 | 139 | 112 | 139 | 27 min |
| &nbsp;&nbsp;same model, station/date hidden | 11 | 133 | 112 | 138 | |
| ResNet50 + station/date, seed 42 | 23 | 148 | 121 | 145 | 45 min |
| &nbsp;&nbsp;same model, station/date hidden | 23 | 148 | 122 | 145 | |
| ResNet50 + station/date, seed 7 | 22 | 145 | 116 | 136 | 45 min |
| &nbsp;&nbsp;same model, station/date hidden | 17 | 135 | 112 | 135 | |
| ConvNeXt-Tiny + station/date (batch 24) | 18 | 142 | 93 | 132 | 52 min |
| &nbsp;&nbsp;same model, station/date hidden | 16 | 127 | 80 | 127 | |

What this shows:

* **Retraining noise is large.** The same ResNet18 setup found 114-130 burst
  files at 2% FA depending on the seed. A difference under about 8 files between
  two single runs means nothing.
* **The deeper backbones do not detect better on this data.** Averaged over
  seeds, ResNet18 found 119 / 140 (2% / 5% FA) and ResNet50 118.5 / 140.5, at
  2.5 times the training time and GPU memory; ResNet34 sat at the low end of the
  ResNet18 range and ConvNeXt-Tiny clearly below it (93 / 132). ResNet18 stays
  the default; the others are there to re-test as the labelled set grows -- with
  about 1,300 burst files, a larger network has little more to learn from.
* **Station/date gives the bounded dependency it is designed to, but no
  measurable detection gain for a good image model.** With it on vs hidden in
  the same ResNet, at equal false alarms: +2, -2, 0, -1, +4 burst files at 2% FA;
  -1, +1, +1, 0, +1 at 5%. Only the weakest image model, ConvNeXt-Tiny, gained
  clearly (+13 / +5). At the operating point it changes 6-17 of 524 verdicts,
  mostly by raising scores slightly at stations with many labelled bursts (BIR,
  GREENLAND, ALASKA, SSRT) -- the file-selection signal the cap is there to
  contain. It is on by
  default because the station's own interference is real signal that more labels
  per station should let it learn; switch it off in the Train tab if you prefer.
* The verdict counts at the calibrated threshold move with where calibration
  landed (19-23 false alarms on test for a 5% validation budget = 5.7-6.9%), so
  compare models on the fixed-FA columns.

### Calibrated to a false-alarm budget

After training, the model is run over the **validation files** exactly as the
Predict tab runs it (`core/file_eval.py`), and the burst threshold is chosen to
find **as many burst files as possible while flagging at most 5% of quiet
files**, then the **fewest false alarms at that recall**, then the middle of the
range doing both. (The first version took the lowest threshold within budget;
recall is flat over most of the range on real files, so that spent the budget for
nothing.) The threshold, the budget, the measured rates and the region-finder
settings it was tuned with are stored in the checkpoint. Predict uses them, and says so if its settings
drift from the calibrated ones. Evaluate reports, per file: false alarms on quiet
files (by station), burst files found -- counted only when the detection lands on
a drawn burst -- and stray detections in burst files. The false-alarm and missed
files are listed first and open in the Label tab on double-click.

Model selection during training uses `unified_score`: the mean of **detection
average precision** (does burst evidence rank bursts above background and RFI)
and **typing macro-F1** among real bursts. Background and RFI keep full loss
weight; only the burst types are balanced among themselves.

### Type frequencies

Non-bursts are by far the commonest thing in the data, and among bursts CALLISTO
records Type III about 86% of the time, Type IV about 10.5%, Type II 2-3% and
everything else about 1%. The training set looks nothing like that, twice over:
rare types get labelled out of proportion (the previous archive was 18% Type II),
and the loss then balances the types further so the rare ones are learned at all.
Both are right for learning, but they leave a model that expects a Type II almost
as often as a Type III -- and with Type III thirty times commoner in the sky, even
a small share of Type III called Type II outnumbers the real Type II.

**Non-bursts** are handled by the operating point, not by the type prior: about 3
background regions per burst sample are mined, background keeps full loss
weight, and the threshold is calibrated to a false-alarm rate *per quiet file*
(≤5%), which holds however rare bursts are.

**Burst types** are corrected after training (`core/type_priors.py`). Each
region's type probabilities are multiplied by *(observed share / training
share)^strength* and renormalised over the burst types, where the training share
is what the model was fitted to -- each class's training samples times its loss
weight. The burst types keep their total, so **burst evidence and the calibrated
threshold do not move**: only *which* type a burst is called changes. The
strength (0 = as trained, 1 = full real-world odds) is chosen on the validation
regions by the **real-world macro-F1** -- each type's F1 with precision
computed as if the types occurred at the observed frequencies. That rewards
calling the commoner type when unsure without letting a rare type vanish, which
plain accuracy at those frequencies would allow (always answering Type III scores
86%). The curve is flat near its top -- far inside the noise of a few dozen
rare-type files -- so, as with the burst threshold, the **middle of the plateau**
is taken: the strengths within one bootstrap standard deviation (over files) of
the best, and the grid value nearest their middle, the gentler on a tie.

Measured on the archive benchmark below (Type IV folded into Other there, since
the old labels have none). The model was trained on a mix of 29.5% Type II,
52% Type III, 18.4% Other, against observed shares of 2.5%, 86% and 11.5%. Typing
of the held-out test regions, scored at the observed frequencies:

| | as trained (strength 0) | corrected (chosen: 0.5) |
|---|---|---|
| Type II found / right when called | 82% / 38% | 72% / **63%** |
| Type III found / right when called | 96% / 96% | 97% / 96% |
| Other found / right when called | 61% / 77% | 72% / 82% |
| real-world macro-F1 | 0.722 | **0.800** |
| real-world accuracy | 91.6% | 93.6% |

As trained, well over half of the regions called Type II would really be
Type III; corrected, about a third are, for one Type II in ten more missed. Every
strength from 0.25 to 1.0 beat the uncorrected model on validation and on test
(test macro-F1 0.749, 0.800, 0.795, 0.767). Be aware that the plateau rule was
settled after this test result was seen: validation's own arg-max was 1.0 (0.768
vs 0.766 at 0.75), and the gentlest-within-noise rule tried first picked 0.25, so
the 0.800 is a best case for this data rather than a clean held-out estimate. The
Predict tab shows the strength and lets you change it.

The observed shares are set in the Dataset tab (*How often each type occurs*) and
recorded in the snapshot's config; Type IIIG is counted within Type III and split
from it by how many boxes of each were drawn, and a folded type adds its share to
the class it trains as. The Train tab logs the training mix beside the observed
one and the chosen strength; Evaluate shows typing at the observed frequencies,
as trained and as corrected; Predict applies the chosen strength and lets you
change it. Regions are counted as events, which is an approximation: a long
Type II yields more candidate regions than a single Type III.

### Measured, per file

A benchmark on the previous label set (2,103 files; it has no Type IIIG or
Type IV boxes, so those were folded, and before the last row its RFI class was
too small and folded as well). Every new row was
trained identically (12 epochs), calibrated on validation files, and judged on
the same 310 held-out test files, never used for training or tuning. Retraining
the same configuration moved each count by about 2, so smaller differences are
noise:

| | False alarms (207 quiet files) | Burst files found (103) | Line-shaped mistakes |
|---|---|---|---|
| previous model, as used (8 regions, argmax) | 6 | 60 | – |
| this version, plain threshold finder | 4 | 71 | – |
| this version, hysteresis finder (two runs) | 6, 8 | 83, 82 | 7 |
| this version, hysteresis + lines kept out of burst classes | 6 | 77 | 0 |
| **+ RFI named automatically, no RFI boxes** (default) | **1** | **79** | – |

The last row changes nothing but the labels of the negatives: the export wrote
the same 22,544 samples, and 10,256 of the 18,669 background ones were named RFI
(8,672 periodic -- mostly the calibration blocks in the lowest channels, which
the finder proposes in many files -- 1,163 carriers, a handful of the rest). The
calibrated threshold rose from 0.72 to 0.86, strays in burst files fell from 14
to 12, and the one remaining false alarm is an ALGERIA-CRAAG file. On the test
regions 7 of 440 burst regions were called RFI and none of the 1,434 RFI regions
a burst. It is one run, but 6 → 1 is well outside the ±2 that retraining the
same configuration moved counts by. (Line-shaped mistakes were not re-counted.)

*Line-shaped mistakes* are thin horizontal regions -- carrier segments, dotted or
not -- called a burst, counted as regions in quiet files and off the drawn boxes
in burst files. They came from the training labels, not the finder: 36% of the
finder regions trained as Type II were line-shaped (1.3% of drawn Type II boxes),
because generous boxes are crossed by carriers, and the model learned "thin
horizontal segment = Type II" -- 10 of 14 validation line mistakes were called
exactly that. No measurement of the segment alone (flatness, slant, continuation
along its channels, persistence across the file) told these from genuinely thin
Type II lanes, so line-shaped regions inside burst boxes are now left out of the
burst classes (`negatives.assign_regions`); the drawn box still teaches its burst.
That removed every line-shaped mistake. Of the 5 burst files it stopped finding,
one was a real faint Type II, one an "Other" box drawn around a dotted carrier,
and three were doubtful detections in interference-heavy files. The Dataset tab
option *Keep carrier-like lines out of burst classes* switches it off -- worth
comparing once your burst boxes are drawn tightly around the bursts themselves.

The remaining mistakes are mostly not interference: regions called a burst in a
burst file but on no drawn box are largely real emission outside the boxes (the
upper band of a Type II, an unboxed Type III), and several quiet-file alarms sit
on burst-like tracks in files labelled *No burst*. Stations with no labelled
files were not measured: a new station's interference is unseen until some of
its files are labelled -- the triage loop below is the quickest way to do that.

### Closing the loop on false alarms

1. Train; read the per-file false alarms on the Evaluate tab.
2. `Assist → Score queue for triage` ranks unreviewed files by the unified model's
   burst evidence, so its likely false alarms come first. Mark them *No burst*;
   their interference then trains as RFI automatically.
3. Re-export with **Mine hard negatives with the latest unified model**: the
   background regions the previous model most wanted to call a burst are taken
   first. Retrain.

This replaces the old two-model cascade, and fixes its central weakness. The
region finder cannot discriminate — measured on this archive, quiet files yield a
*median of 12* bright regions, more than burst files do. Making rejection a class
the model learns turns that from an unfixable problem into a trainable one.
Measured on real data: it rejects **99%** of background regions.

### Burst parameters

Frequency drift rate is what physically separates the burst types: a Type III is
an electron beam at 0.1–0.5c drifting at roughly `-0.01·f^1.84` MHz/s
(Alvarez & Haddock 1973), a Type II a shock front two orders of magnitude slower.
There are two measurements, for two consumers.

**The parameters of a drawn box** -- what the Label tab shows and stores -- are
calculated **from the box**, and **only for Type II and Type III**
(`burst_physics.box_parameters`). Draw the box from the burst's start to its
end: its height is the frequency range and its width the duration. A Type II or
III drifts from high to low frequency, so the burst starts at the top of the box:

| | |
|---|---|
| f_start, f_end | the box's highest and lowest frequency |
| t_start, t_end, duration | the box's first and last time |
| bandwidth | f_start − f_end |
| **drift rate** | **df/dt = (f_start − f_end) / (t_start − t_end)**, negative |
| **relative drift** | `(1/f)(df/dt)` at the box's middle frequency, comparable across bands since drift scales as ~f^1.84 |

They are recalculated whenever a box is drawn, resized or retyped, and boxes
stored by an earlier version are recalculated the first time their file is
opened. Type IIIG (its box spans a group of bursts, not one), Type IV and Other
get no parameters; Type III and IIIG boxes still get the count of separate
bursts inside them, for the IIIG hint. Regions the Predict tab calls Type II or
III report their drift the same way, from their own frequency range and
duration.

On the archive's boxes this gives a median |df/dt| of **1.9 MHz/s for Type III**
(86% inside the published 1–200 MHz/s) and **0.14 MHz/s for Type II** (94% inside
0.05–1 MHz/s), and a value for every box; the earlier pixel fit gave a usable
value for only 801 of the 1,513 Type III boxes. The result is only as good as the
box: drawn wider in time than the burst, it reads slower.

**What the model is given** is different, and unchanged: 8 physics features
measured from the pixels of every *candidate region*, background and
interference included (`burst_physics.measure_burst`). At prediction there is no
drawn box and no type, so these cannot come from a box, and a value present only
for Type II/III samples would teach the model the label itself. Inside the
region the burst is isolated (adaptive threshold at the region's 93rd
percentile, largest connected component), its ridge tracked along whichever axis
it spans more samples of, and the drift fit with Theil–Sen so interference
outliers cannot drag it. A region with no clean track arrives as zeros with a
`measured` flag off, so the model can tell "no drift" from "not measurable".

### What the physics branch is worth

A/B on the same split, unified model with and without the branch:

| | image only | + physics |
|---|---|---|
| macro-F1 | 0.704 | **0.769** |
| burst recall | 0.919 | **0.952** |
| bursts missed | 17 / 209 | **10 / 209** |

### Consistency checking

The Label tab shows the drift beside the assigned type and warns when they
disagree -- a box marked Type II drifting at Type III speed usually means the box
is around the wrong feature, or drawn far too narrow. It never changes a label.

**The warning bounds are wider than the literature** (0.001–2 MHz/s for Type II,
0.02–500 for Type III), to catch gross errors only: a check that fires on
ordinary boxes gets ignored. On the archive's boxes, with the box drift, they
flag 9 of 382 Type II boxes (all drifting faster than 2 MHz/s) and none of 1,513
Type III boxes. The published single-burst ranges stay available as
`LITERATURE_DRIFT_RANGES`.

### Where the training data comes from

| Class | Source |
|---|---|
| Type II / III / IIIG / IV / Other | your drawn boxes, **plus** finder-proposed regions that fall inside them -- unless the region is a carrier fragment (its channels stay bright outside the box) or is line-shaped (≤22 channels tall, ≥3× wider -- see *Measured, per file*). Where boxes overlap, see *Overlapping boxes* below |
| RFI | the background regions below that carry an interference signature, named automatically (see *Interference is found, not drawn*); plus **synthetic interference** (carriers, impulses, periodic pulses, sweeps, gain steps) painted onto a fifth of the quiet files |
| No_Burst | the rest of the finder-proposed regions in confirmed no-burst files and outside the boxes in burst files -- together with RFI about **3 per burst sample**, the regions inference will examine first, hardest first (by the previous model when hard-negative mining is on) |

**Overlapping boxes.** Boxes often overlap -- a Type III crossing a Type II
lane, or a burst deliberately boxed inside a larger one (105 of the archive's
710 burst files). "Inside a box" is therefore measured on a pixel map in which
every pixel belongs to the **smallest** box covering it, and a finder region
takes the type owning at least 60% of it:

- where boxes of two types overlap, the **smaller box's type** wins: a Type III
  drawn on a Type IV continuum is Type III there and Type IV elsewhere;
- boxes of the **same type** count together, so a region straddling two
  overlapping Type III boxes is Type III even if neither holds 60% of it;
- a region inside the boxes but **split between two types**, none owning 60%, is
  left out;
- a background region must lie at most 10% on *all* the boxes together, and a
  detection counts as landing on a burst by the same union.

The drawn boxes are still cropped exactly as drawn, overlap included, since at
prediction bursts sit together too. On the 45 readable archive files with
overlapping boxes this changed 43 of 1,928 region labels: 32 regions that were
thrown away are now Type III, 5 moved from Other to the Type III drawn inside
the Other box, 3 split between two types and 3 background regions that were
over 10% on bursts are left out. The Dataset tab reports these counts after each
export. That is about 0.2% of a whole snapshot, well inside what retraining
moves the file-level counts by, so it was not re-benchmarked: it makes the
labels right, not the headline numbers different.

Each snapshot also writes `files.csv`: every file, its verdict and split, which
is what file-level calibration and evaluation run on. The split is made over
*files* (stratified by verdict, grouped by solar event), not over samples, so it
does not move when mining or finder settings change: two snapshots of the same
labels are judged on the same test files.

The first two rows matter, and the positive side is the subtle one.

**Positives must be finder-shaped too.** An earlier version used only hand-drawn
boxes as positives while every negative came from the region finder. The two
classes then differed in *how the region was produced*, not just in what it
contained — and the model learned that shortcut. It scored 0.95 burst recall on
held-out crops and detected **1 burst file in 12** at inference, because every
finder region looked like a negative. Labelling finder regions by their overlap
with drawn boxes puts both classes on the same footing, exactly as an object
detector assigns anchors. After the fix: **22 of 25** burst files, **7 of 7**
quiet files.

**Overlap is measured by containment, not IoU.** A drawn box averages ~46,900 px
here while a candidate region averages ~388 px — 121× smaller. IoU between them
is ~0.003 even when the candidate is *entirely inside* the burst, so an IoU rule
discarded 85% of genuine matches (72 matched regions, versus 603 with
containment). Containment asks the question that actually matters: is this region
inside a burst.

Matched positives are capped per drawn box (default 2) so one generous annotation
cannot flood its class with near-duplicates of a single event.

### Measured on real annotations (previous version, crop level)

These are the v1 numbers, kept for the record. They are crop-level, which is
exactly what hid the false-alarm problem; judge a model by the per-file report.

490 drawn boxes + 46 no-burst files → 2,089 samples, split 996 No_Burst / 1,093
burst crops, zero event leakage:

| | value |
|---|---|
| test accuracy | 0.812 |
| macro-F1 | 0.734 |
| burst recall | **0.938** |
| burst precision | 0.985 |
| background rejected | **0.982** |
| type correct, given detected | 0.724 |

The Evaluate tab reports that burst-vs-background rollup alongside the multiclass
metrics, because "did it notice a burst" matters more operationally than "did it
name the right type": confusing Type II with Type III is recoverable, calling a
real burst background is not.

## The legacy two-model track

Still selectable everywhere, so earlier checkpoints stay reproducible and you can
A/B against the unified model on the same data.

| | Burst type | Burst / no burst |
|---|---|---|
| Sample | one per drawn box (crop) | one per file (whole spectrum) |
| Classes | Type II / Type III / Other | Burst / No_Burst |
| Head | 3-way softmax | single logit + tuned threshold |
| Metadata branch | no — morphology, not station, separates types | yes — station, frequency range, cyclical date |
| Selection metric | macro-F1 | PR-AUC |

Splits are **event-grouped**: every crop from one file, and every station's
recording of one solar event, lands in the same split. Without that, two boxes
cut from the same spectrum could end up in train and test, and the reported score
would be inflated by a model that had already seen the answer. The exporter
reports `event_leakage`, and the Train tab refuses to start if it is non-zero.

---

## Frequency axis: two separate bugs, both fixed

### 1. The axis table is not always named

The reader located the axis table by `EXTNAME='AXES'`. A large part of the
archive writes the **identical table with no `EXTNAME` at all**, so those files
were skipped and fell back to the header. The result was a "frequency axis" that
was simply the channel index — **1–200 MHz for a 200-channel receiver** —
regardless of the band actually observed.

The table is now found by **structure**: any extension carrying `TIME` and
`FREQUENCY` columns whose lengths match the image. A named `AXES` table is still
preferred, so files that already worked are unaffected. Sampled across the
archive afterwards, **60 of 60 files resolve a real axis**; none fall back.

| File | Was shown | Actually |
|---|---|---|
| `ALASKA-ANCHORAGE_..._01` | 1 – 200 MHz | **5.00 – 65.88 MHz** |
| `MEXART_..._59` | 1 – 200 MHz | **46.00 – 225.94 MHz** |
| `Australia-ASSA_..._63` | 1 – 400 MHz | **15.00 – 88.37 MHz** |

**This affected measured physics, not just the display.** Drift rate is MHz per
second, so it is only meaningful once a row is worth the right number of MHz.
`Reset → Recheck frequency axes...` re-reads every file and re-measures the bursts
whose axis changed; boxes, types and verdicts are never touched. On this dataset
it corrected 6,000 files and re-measured 151 bursts in 52 seconds.

### 2. The header keywords are placeholders anyway

The upstream reader derived the frequency range from the `CRVAL2`/`CDELT2` header
keywords. **In this archive those are placeholders** (`CRVAL2` ≈ 193–200,
`CDELT2` = −1) and do not describe the real coverage. Every one of 12 sampled
files across 10 stations disagreed with its true axis; several produced
physically impossible *negative* frequencies.

| File | Header says | Actually |
|---|---|---|
| `ALASKA-ANCHORAGE_20230613_2301_2306` | 20 – 200 MHz | **5.9 – 65.9 MHz** |
| `GERMANY-DLR_20240724_0731_0742` | −716 – 200 MHz | **11.4 – 1594.9 MHz** |

The true axes live in an `AXES` BinTable extension (`TIME` in seconds,
`FREQUENCY` in MHz, stored **descending** so array row 0 is the *highest*
frequency). It is present in about half the archive — the 5-minute
`STATION_DATE_HHMM_HHMM.fit.gz` files carry it, the 15-minute
`STATION_DATE_HHMMSS_01.fit.gz` files do not.

This project prefers the `AXES` table, falls back to the header, and records
which was used in `freq_axis_source`. Files on the fallback are flagged in the UI
as *"frequency axis approximate"*. The old header-derived values are still stored
as `legacy_freq_min_mhz` / `legacy_freq_max_mhz` so a model can be trained either
way and compared.

**This changes the metadata branch of the binary model.** Models trained here are
not directly comparable to existing Burst Identifier checkpoints unless the fix
is ported upstream. The spectrum tensors themselves are unaffected.

---

## Performance

The labelling loop is navigation-bound: decoding a gzipped spectrum costs
100–400 ms, and files range from `173×1680` to `177×39600` samples.

- **Look-ahead prefetch** — displaying file *i* decodes *i+1…i+4* on a thread
  pool, so Next is instant.
- **Two-level cache** — a byte-bounded in-memory LRU, plus normalized arrays
  persisted as `.npy` keyed by file content *and* preprocessing settings, so a
  second pass through the queue skips the gzip entirely.
- **Level-of-detail rendering** — only the visible column span is uploaded,
  max-pooled to at most 4,000 columns. **Max**, never mean: a Type III lane can
  be two columns wide and averaging would erase it. Zooming in re-renders the
  smaller span at full resolution. Crops always come from the undecimated array.
- **Virtualized queue** — the list model holds only integer ids; row content is
  fetched on demand, so 100k+ files load instantly.
- **Training runs out-of-process** — CUDA plus a Qt event loop is a reliable way
  to produce hangs, a crash cannot take the window down, and labelling continues
  while a model trains.

---

## What you see is what the model gets

The canvas defaults to exactly the `−1 … 8 dB` window the training tensor uses.
Contrast can be stretched to hunt for faint features, but a badge lights up
whenever the view differs from the model's, because a burst that is only visible
under a stretched window is one the model has no chance of learning.

The right panel renders the **actual 224×224 tensors** the selected box will
produce — the crop, the context view and the quiet-background context, with the
same normalization, margin and interpolation as the exporter.
`tests/test_ui_label.py` asserts the preview is byte-identical to the exported
sample.

---

## Exporting a trained model

`Evaluate → Export model...` writes a self-contained bundle:

| File | Purpose |
|---|---|
| `model_card.json` | the full input contract, classes, threshold, metrics, provenance |
| `weights.pt` / `checkpoint.pt` | state dict, and the original checkpoint |
| `config.yaml` | the training configuration |
| `model_scripted.pt` | TorchScript graph — runs with no project code at all |
| `predict.py` | standalone script using only numpy, astropy and torch |
| `README.md` | how to use it, and what it must not be used for |

A bare `best.pt` is not enough to use a model correctly: it carries weights but
not the knowledge that its input must be background-subtracted over the whole
file, mapped through a −1…8 dB window, resized to 224×224 and — for the type
model — cropped from a *region* with a 15% margin. Getting any of that wrong
produces confident, wrong answers rather than an error. The card records all of
it, and `tests/test_model_export.py` asserts the standalone script reproduces the
project's preprocessing **bit-for-bit**.

```bash
python predict.py path/to/file.fit.gz
```

## Prediction on new files

The Predict tab runs a **region-based cascade**, mirroring how the models were
trained:

1. the binary model scores the **whole file**;
2. if it says burst, bright regions are located in the normalized spectrum;
3. each region is cropped **exactly as the exporter crops a drawn box**;
4. the type model classifies each crop.

Step 3 is the reason this is not simply the upstream `predict_file()`. That
function runs the type model on the whole-file tensor, which was right when the
type model was trained on whole files — but this app's type model learns from
crops, so feeding it a whole file is a train/serve mismatch. It would answer
confidently anyway.

### The binary gate is not optional

Measured on this archive (20 confirmed burst files across stations, 10 confirmed
`No_Burst` files), the region finder produced at least one candidate in **10 of
10 quiet files at every threshold from 0.22 to 0.45**, with a median of 12
regions — *more* than the 7 median in burst files. Bright connected regions are
everywhere: interference, carrier lines, calibration artifacts.

So candidate regions carry **no evidence** that a burst occurred; only the binary
model does. Running the type stage without the binary gate produces typed regions
for completely quiet files, and the tab says so in an explicit warning.

### Adaptive brightness

Stations differ enormously in gain, so the same event peaks at 1.0 in one
recording and 0.43 in another. With "Adapt to each file" on (the default), the
brightness setting acts as a ceiling and each file's own bright tail lowers it
when needed. Measured over 20 burst files this lifts recall from 19/20 to 20/20
with no increase in regions found on quiet files.

### Faint bursts that fragment: hysteresis

Adapting cannot help a faint burst in a file that also holds bright interference:
the interference pins the file's level at its ceiling, and the burst -- a lane at
+1.3 to +1.8 dB -- crosses it only in scattered patches, each under the 60-pixel
minimum. 95 of 710 labelled burst files (13%) were reachable only that way.
(`ALASKA-COHOE_20230617_2243_2248`, a clearly visible Type III that breaks into
sub-60-pixel pieces, was the first one noticed.)

So the finder uses **hysteresis** (`core/region_finder.py`): pixels at the level
are seeds, and each seed's region grows through the connected pixels of a lightly
smoothed copy that reach 0.65 × the level. A region needs 5 seed pixels, so faint
structure with no bright core never becomes a candidate, and its box is measured
over its own pixels, so smoothing never widens it.

| Finder, on the train + validation files | Burst files reached | Lost to merging |
|---|---|---|
| plain threshold | 82.5% | 1.5% |
| **hysteresis** | **90.1%** | 1.5% |
| (min area 20, all files) | 83.2% | 1.5% |
| (dilating the mask, all files) | 77–79% | 9–13% |

Lowering the minimum area adds specks everywhere; dilating glues bursts to the
interference beside them. The setting is recorded in each checkpoint, so a model
always proposes regions the way its training regions were mined. It is still a
heuristic with real limits, not a detector.

## Reading the binary metrics honestly

Burst datasets are usually positive-heavy, and that quietly breaks the headline
numbers. On a real 281-Burst / 46-No_Burst set the model reported **F1 0.911** on
test — which sounds strong, but *always answering "Burst"* scores **0.925** on
that same split. The model was below a constant predictor and nothing said so.

The Evaluate tab therefore always reports, next to F1:

- **how many real bursts were missed** (`7 of 43 (16%)`), which is the number that
  actually matters for this science;
- the **always-answer-majority baseline**, with an explicit "above / at or below"
  verdict;
- **balanced accuracy**, which collapses to 0.5 for a constant predictor and so
  cannot be inflated by imbalance.

The Dataset tab warns before you train when classes are more than 3:1 apart, or
when the minority class has under 50 samples.

### The decision-threshold bug this exposed

Threshold tuning originally swept a fixed `linspace(0.05, 0.95)`. That floor
silently capped the search. A well-separated but poorly calibrated model can put
every negative at *exactly* 0.0 while still scoring real bursts at 0.00015 — so
the whole useful range lies **below 0.05**, and `f1`, `accuracy`, `recall` and
`balanced_accuracy` all returned the identical clipped answer of 0.05.

The consequence was exactly the "burst files come out as No_Burst" symptom:

| | threshold | test F1 | bursts missed | false alarms |
|---|---|---|---|---|
| fixed 0.05–0.95 grid | 0.050 | 0.9114 | **7 of 43** | 0 |
| data-derived search | 0.000077 | **0.9767** | **1 of 43** | 1 |

`find_best_threshold` now evaluates the midpoints between consecutive observed
probabilities — the only points where predictions actually change — so the
optimum is reachable at any scale. Candidates that would put every sample in one
class are dropped, which is what the old numeric bounds were really guarding
against, checked by outcome rather than assumed.

A threshold far from 0.5 is now reported as a **calibration** note rather than
treated as normal: the model separates the classes, but its probabilities cannot
be read as confidences.

**If you trained before this fix, retrain.** The weights are fine; only the stored
threshold is wrong, and retraining re-tunes it automatically.

## Assisted pre-labelling

Two capabilities, kept distinct because they carry very different trust:

- **Triage ordering** (`Assist → Score queue for triage`) runs a trained binary
  checkpoint over unreviewed files and sorts the queue by burst probability. A
  real model prediction; the only claim made is a ranking. No label is written.
- **Box suggestions** (`P`) are **not a trained detector**. The type model is a
  classifier — given a region it names the type, but it has no notion of *where*
  a burst is. Candidate regions therefore come from a plain brightness threshold
  plus connected-component grouping, and the type checkpoint classifies each one.
  It will flag RFI and instrument artifacts as readily as bursts. Suggestions are
  stored unconfirmed, drawn dashed, and **never exported** until you give one a
  type.

---

## Relationship to Burst Identifier

The pipeline from `H:\Burst Identifier` is **vendored** into
`callisto_trainer/core/` — copied, with imports rewritten and numerics untouched.
This project is therefore self-contained and runs without the H: drive.

The risk of vendoring is silent drift, so `tests/test_preprocess_parity.py`
imports the *original* modules directly from the H: tree and asserts the vendored
copy still produces bit-identical output, on synthetic data and on real files.
Those tests skip cleanly when the drive is absent.

Deliberate deviations, all documented in the source:

- the frequency-axis fix described above;
- `DATE-OBS` slash-separated dates are now parsed (the original silently fell
  back to the filename);
- `emit_progress` calls in the two trainers for live GUI curves;
- `logging.py` renamed to `logging_utils.py` to remove any stdlib ambiguity.

---

## Layout

```
callisto_trainer/
  core/       vendored pipeline + crops/coords/inference — no Qt, headless-testable
              taxonomy.py        the labels and how they relate
              region_finder.py   candidate regions (shared by training and inference)
              region_features.py interference features; region_inputs.py builds model inputs
              crops.py           crops, the context view and the quiet background
              rfi_labels.py      names the interference among the rejections RFI
              synthetic_rfi.py   interference painted onto quiet files
              type_priors.py     corrects the burst type for how often each type occurs
              file_eval.py       per-file evaluation, false-alarm and type calibration
  store/      SQLite schema, repository, dataset export
  services/   import, caching, prefetch, training subprocess, assist, model export
  ui/         PySide6 widgets
configs/      trainer.yaml (app paths and responsiveness)
data/         annotations.db, display cache
datasets/     exported snapshots (immutable)
outputs/      checkpoints, reports, figures, exported model bundles
```

`core/` is Qt-free by design: every scientific function is unit-testable headless
and runnable from the command line, e.g.

```bash
.venv\Scripts\python -m callisto_trainer.core.train_type --config datasets/types/<run>/config.yaml
```

---

## Tests

```bash
.venv\Scripts\python -m pytest tests/ -q
```

526 tests. The ones that matter most:

| File | Guards |
|---|---|
| `test_crop_equivalence.py` | full-extent crop is bit-identical to `preprocess_array`; crop order actually preserves burst signal |
| `test_preprocess_parity.py` | the vendored copy has not drifted from the H: original |
| `test_axes.py` | AXES-table extraction, header fallback, the named 20–200 vs 5.875–65.875 regression |
| `test_export.py` | no source file and no solar event spans two splits |
| `test_store.py` | labels survive a simulated process kill; concurrent reads during writes |
| `test_ui_label.py` | crop preview equals the exported tensor; session resume |
| `test_assist.py` | unconfirmed suggestions are never exported |
| `test_raw_and_reset.py` | raw view keeps NaNs the model never sees; resets scoped correctly; backup is restorable |
| `test_model_export.py` | the bundle's standalone script reproduces preprocessing bit-for-bit; TorchScript matches eager |
| `test_inference.py` | stage 2 receives a **crop**, stage 1 the **whole file**; the gate blocks typing on no-burst files |
| `test_v2_pipeline.py` | each interference feature separates its pattern; calibration holds its false-alarm budget; RFI never counts as a detection; there is no RFI box; export → train → calibrate end to end |
| `test_quiet_view.py` | a long continuum stays bright in the quiet background and a short burst looks the same; the third view is built identically by export, prediction, the bundle script and the Label tab; a three-view model refuses to run without it |
| `test_burst_parameters.py` | a Type II / III box's drift is (f_start − f_end)/(t_start − t_end) from its top-left to bottom-right; other types get none; drawing, resizing, retyping and reopening recalculate; predicted Type II / III regions report the same |
| `test_overlapping_boxes.py` | the smallest box owns a shared pixel; a burst inside a bigger box takes its type; same-type boxes count together; a region split between two types is left out; negatives and detections use all the boxes at once |
| `test_rfi_reporting.py` | a file with a burst and RFI is Burst and lists the RFI; a file with only RFI is No_Burst; RFI and No_Burst are one class in reports; segments on the same channels are one interference source; deleting over 2 GiB of snapshots reports it |
| `test_type_priors.py` | each interference signature is named RFI and a drifting burst is not; the type correction never changes burst evidence, turns an unsure burst into the commoner type and leaves a confident rare one alone |

Tests needing real FITS files use the archive on `H:` and skip if it is absent.
