# Deepfake Image Detection

Binary classifier that separates authentic face photographs from digitally
manipulated ones, built on MobileNetV2 transfer learning with TensorFlow/Keras.

The project ships two independent interfaces: a self-contained Jupyter notebook
for reading and demonstration, and a tested Python package with a CLI for
experiments and automation. Evaluation follows a sealed-test protocol: every
headline number is reported against the majority-class baseline with a
bootstrap confidence interval, and the operating threshold is chosen on the
validation split only.

---

## Just run it

1. Open `E:\...\Deepfake Image Detection\`
2. Double-click **`open.bat`**
3. In Jupyter: **Cell → Run All**

That opens **`Deepfake Detection.ipynb`**, which is self-contained: it defines every
function it uses, imports nothing from `src/`, and needs no `PYTHONPATH` and no
command line. It runs top to bottom in one pass: data, EDA, split, pipeline, model,
training, fine-tuning, evaluation, Grad-CAM, inference, save.

It ships already executed, so the outputs and figures are visible before you run
anything. Takes about 8 minutes on CPU to re-run.

Everything below is optional. The same logic also exists as a tested package with a
CLI, for running experiments without a browser. **You never need both** — pick one.

---

## Quick start (package + CLI)

```bash
# 1. Environment (Python 3.11 required: TensorFlow 2.15 has no 3.12+ wheels)
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. Dataset. Either download it from Kaggle:
#    https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection
#    and extract so that data/real_and_fake_face/{training_real,training_fake}
#    exists, or point data.root in config.yaml at wherever you put it.

# 3. Everything else
set PYTHONPATH=src
python -m deepfake.cli smoke        # 30-second end-to-end check
python -m deepfake.cli all          # split, train, evaluate, gradcam
```

Or on Windows just run `launcher.bat` and pick from the menu.

## Commands

| Command | What it does |
|---|---|
| `python -m deepfake.cli split` | Builds the stratified train/val/test manifest at `artifacts/splits/manifest.csv` |
| `python -m deepfake.cli info` | Prints split sizes, class balance and the majority-class baseline |
| `python -m deepfake.cli train` | Two-stage transfer learning, writes `artifacts/<run>/model.keras` |
| `python -m deepfake.cli evaluate` | Sealed-test evaluation, writes `metrics.json` and `evaluation.png` |
| `python -m deepfake.cli gradcam` | Attention figure for a balanced sample of test images |
| `python -m deepfake.cli predict a.jpg b.jpg` | Classifies image files |
| `python -m deepfake.cli all` | split, train, evaluate, gradcam in sequence |
| `python -m deepfake.cli smoke` | One tiny epoch end to end: proves the wiring, not the accuracy |
| `pytest` | The test suite (57 tests, no GPU, no dataset needed) |

`--config path/to/other.yaml` works on either side of the subcommand.

## Running an ablation

Never hand-copy `config.yaml`. Derive the variant, so the diff against the
baseline is exact and the file says what it changed:

```bash
python experiments/derive.py res384 image.size=[384,384]
python -m deepfake.cli --config experiments/res384.yaml train
python -m deepfake.cli --config experiments/res384.yaml evaluate
python experiments/compare.py      # every run, sorted by threshold-free AUC
```

`compare.py` reads only `metrics.json`, so every number it prints was measured.
It sorts by ROC AUC rather than accuracy, because accuracy moves when only the
operating point changes (DECISIONS.md D-020), and it prints the confidence
intervals so you can see when a difference is not established.

## The two interfaces

| | `Deepfake Detection.ipynb` | `src/deepfake/` + CLI |
|---|---|---|
| For | reading, demoing, a single reproducible run | ablations, tests, automation |
| Needs | `open.bat` and a browser | `PYTHONPATH=src`, a terminal |
| Config | the `CONFIG` cell inside it | `config.yaml` |
| Output | `artifacts/notebook/` | `artifacts/<run_name>/` |
| Tests | none (it is the demo) | 57 tests |

They are independent implementations of the same method and neither imports the
other, so the notebook can never break because of a refactor in `src/`. The cost is
that a change to the method has to be made in both; the benefit is that the notebook
is genuinely standalone, which is the whole point of it.

## Layout

```
config.yaml            every tunable value; nothing is hardcoded in src/
src/deepfake/
  config.py            load + validate config, resolve paths
  data.py              discovery, stratified split manifest, tf.data pipeline
  model.py             backbone + head; preprocessing and augmentation as LAYERS
  train.py             two-stage training with explicit callback directions
  evaluate.py          sealed-test metrics, bootstrap CIs, calibration
  predict.py           inference against the saved deployment contract
  gradcam.py           attention maps taken from the logit
  cli.py               one entry point for all of the above
tests/                 57 tests; each pins a specific behavioural guarantee
experiments/
  derive.py            generate a config variant by dotted override
  compare.py           table of every finished run, read from metrics.json
Deepfake Detection.ipynb   the standalone notebook; all logic inline
artifacts/<run_name>/  model.keras, label_map.json, metrics.json, figures, logs
data/                  the raw dataset (not tracked)
everything_u_need_to_know.pdf   complete project walkthrough and interview guide
```

## What the model is

```
input  (H, W, 3) raw uint8 pixels in [0, 255]
  -> augmentation      horizontal flip, mild rotate/zoom/translate/contrast   (training only)
  -> preprocess        Rescaling(1/127.5, offset=-1)  ->  [-1, 1]
  -> MobileNetV2       ImageNet weights, include_top=False, BatchNorm always inference-mode
  -> GlobalAveragePooling2D
  -> Dropout
  -> Dense(1, linear)  a LOGIT, not a probability
```

Training uses `BinaryCrossentropy(from_logits=True)`. Inference applies the
sigmoid explicitly, so `sigmoid(logit) = P(real)`.

Stage 1 trains the head with the backbone frozen. Stage 2 unfreezes the top 40
backbone layers at a 10x lower learning rate, keeping every BatchNormalization
layer frozen. Each stage checkpoints separately and the genuinely better one
becomes `model.keras`.

## Reading the results

`artifacts/<run>/metrics.json` reports, for the sealed test split:

- accuracy **with the majority-class baseline next to it** and a bootstrap 95 %
  confidence interval, because on ~300 images the interval is wide enough to
  change how a result should be described
- ROC AUC and PR AUC for both classes
- log loss, Brier score, and a calibration curve with expected calibration error
- the confusion matrix, per-class precision/recall/F1
- accuracy broken down by CIPLAB difficulty tier (easy / mid / hard)
- how the model performs if uncertain cases abstain to human review

Metrics are chosen for a binary probabilistic classifier: threshold-free
ranking quality (ROC AUC, PR AUC), probability quality (log loss, Brier,
calibration), and thresholded performance always alongside its baseline.

## The dataset

CIPLAB `real-and-fake-face-detection`, 2,041 images: 1,081 real and 960 fake.
The fakes are **expert Photoshop composites, not GAN output**, so the artefacts
are blending and lighting inconsistencies rather than generator fingerprints. A
model trained here should not be expected to transfer to GAN-generated faces.

Fake filenames encode useful metadata that the manifest parses and keeps:
`easy_100_1111.jpg` means difficulty `easy` and a region mask of
`1111` = left eye, right eye, nose, mouth all manipulated.

## Limitations and scope

- Accuracy is modest. A frozen ImageNet backbone at 224x224 is architecturally
  weak for image forensics: forgery cues are high-frequency and local, while
  downsampling and global average pooling remove exactly that. Measured numbers
  live in `artifacts/<run>/metrics.json` and are summarised in CONTEXT.md.
- The split is stratified by class but not by source identity. CIPLAB fakes are
  built from the real photographs in the same collection, so some identity
  overlap across splits is possible and the numbers may be slightly optimistic.
- Single dataset, single manipulation family. Cross-dataset evaluation
  (FaceForensics++, Celeb-DF) is the definitive test and has not been run.
- **This must not be used as evidence about a real person.** At this accuracy a
  positive result means "worth a human look", nothing more.

See DECISIONS.md for the reasoning behind each design choice and ARCHITECTURE.md
for how the pieces fit together.
