# Decision log

Append-only. A superseded decision gets a new entry; it is never rewritten.
Each entry records the decision, the context that forced it, and what it costs.

---

## D-001 Configuration lives in one YAML file, and the code validates it

**Decision.** Every tunable value lives in `config.yaml`. `config.py` validates
it at load time and refuses to run on an invalid file. No module hardcodes a
number, a path or a threshold.

**Context.** Hyperparameters scattered across modules or notebook cells drift:
the same value ends up defined twice with different numbers, and hardcoded
absolute paths break silently the moment a project moves. Centralising and
validating the configuration removes both failure modes.

**Consequences.** One file to read before a run and one file to diff after it.
`config.used.yaml` is copied into every run directory, so any result traces back
to the exact settings that produced it. Cost: a layer of indirection, and a
config schema that must be kept in step with the code (the tests do that).

---

## D-002 The split is materialised to a CSV manifest, not recomputed

**Decision.** `build_manifest` writes `artifacts/splits/manifest.csv` once,
stratified per class and seeded. Every command reads it.

**Context.** Recomputing a split in several places (for example, repeated
`image_dataset_from_directory(validation_split=...)` calls) relies on every call
site passing identical seeds and parameters. Any mismatch produces a silently
different split, and there is no way to inspect what landed in which subset.

**Consequences.** The split is auditable, reproducible and inspectable.
Disjointness is asserted on creation. It also gives a natural home for per-image
metadata (difficulty tier, region mask). Cost: `split` must be re-run with
`--force` after changing the split fractions.

---

## D-003 Three-way split with a sealed test set

**Decision.** 70 / 15 / 15 train / val / test. The threshold and every model
selection decision use `val`. `test` is read once, by `evaluate`.

**Context.** With only train and validation, the same validation images drive
early stopping, checkpoint selection and the headline number, which makes the
reported figure optimistically biased by construction. A sealed test split is
the minimum protocol for an unbiased generalisation estimate.

**Consequences.** The reported number is a genuine generalisation estimate.
Cost: ~300 fewer training images and a smaller validation set, on an already
small dataset. On 2,041 images, stratified k-fold would be statistically better;
a sealed split was chosen because it is simpler to reason about and because
model selection integrity mattered more than a few points of variance.

---

## D-004 Preprocessing is a layer inside the model

**Decision.** `Rescaling(1/127.5, offset=-1)` sits inside the Keras model. The
data pipeline and the inference helper both hand it raw `[0, 255]` pixels.

**Context.** MobileNetV2's ImageNet weights were trained on `preprocess_input`
output in `[-1, 1]`. If normalisation lives in the pipeline, every consumer
(training pipeline, inference helper, any future serving path) must carry its
own copy of the arithmetic, and copies drift: feeding `[0, 1]` instead evaluates
every activation in the network off-distribution.

**Consequences.** One implementation, carried by the saved model, so training and
serving cannot disagree. Swapping the backbone swaps the rescaling through the
`PREPROCESSING` table. Cost: the saved model is backbone-specific about its
input range, which is why `label_map.json` documents it explicitly.

---

## D-005 Augmentation is a layer, not a pipeline stage

**Decision.** The augmentation `Sequential` is part of the model and therefore
runs per batch and switches off at inference.

**Context.** In a pipeline of the form
`map(preprocess).map(augment).cache().shuffle().prefetch()`, the `cache()` after
`augment` materialises the first epoch's random transforms and replays them
identically forever, so augmentation contributes exactly one extra fixed view of
the dataset instead of a fresh view each epoch. Keeping augmentation inside the
model removes that ordering trap entirely.

**Consequences.** Caching the decode step is now safe and genuinely fast.
Augmentation cannot be accidentally frozen. Cost: the transforms run on whatever
device the model runs on rather than on the CPU input thread.

---

## D-006 Augmentation is conservative and forensics-aware

**Decision.** Horizontal flip plus mild rotation, zoom, translation and contrast.
No vertical flip, no brightness jitter, no aggressive photometric noise.

**Context.** Faces are never upside down at inference, so a vertical flip
teaches invariance to something that does not occur; and photometric jitter
attacks precisely the low-level statistics (noise consistency, blending seams,
compression history) that distinguish a composite from an original. Aggressive
generic augmentation is counterproductive for image forensics.

**Consequences.** Less regularisation than a generic vision task would use, on
purpose. The better forensic augmentation is random JPEG re-compression, which is
listed as future work rather than implemented, because it needs a decode-encode
round trip inside the graph.

---

## D-007 BatchNormalization stays in inference mode, always

**Decision.** The backbone is called as `backbone(x, training=False)`, and during
stage 2 every `BatchNormalization` layer is explicitly left `trainable=False`.

**Context.** Unfreezing backbone layers also unfreezes their BN layers by
default, which then begin adapting ImageNet moving statistics, estimated over a
million images, to a dataset three orders of magnitude smaller. That destroys
the very statistics that make the pre-trained features work.

**Consequences.** Fine-tuning nudges the convolutional filters without destroying
the normalisation statistics. Cost: BN cannot adapt to a genuinely different
input distribution; for a much larger target dataset this decision should be
revisited.

---

## D-008 The output layer is linear; the loss uses `from_logits=True`

**Decision.** `Dense(1, activation=None)` and
`BinaryCrossentropy(from_logits=True)`. `predict.py` applies the sigmoid.

**Context.** Two reasons. Numerical stability: fusing the sigmoid into the loss
avoids `log(0)`. And attribution quality: Grad-CAM on the post-sigmoid
probability differentiates a function whose gradient vanishes once the sigmoid
saturates, so confident predictions produce noise heatmaps. Differentiating the
logit keeps the attribution meaningful at any confidence.

**Consequences.** Grad-CAM is meaningful at any confidence. Cost: the raw model
output is not a probability, which is a real foot-gun for anyone loading
`model.keras` directly, so `label_map.json` states `"output": "logit"` and
`Predictor` raises if that field says otherwise.

---

## D-009 Callback directions are declared, never inferred

**Decision.** `training.early_stopping.mode` and `training.checkpoint.mode` are
**required** config keys. `config.py` raises if either is missing.

**Context.** Keras' default `mode='auto'` infers direction from the metric
*name*: names containing `acc` are maximised, everything else is minimised.
`val_auc` contains `auc`, not `acc`, so under the default AUC is **minimised**:
the checkpoint saves the worst epoch and `restore_best_weights` restores it.
This is one of the most damaging silent defaults in the Keras API, and it can
push a tuned model below chance without a single error message.

**Consequences.** The failure mode is impossible to introduce silently: a
config without an explicit mode does not load, and
`tests/test_config.py::test_callback_mode_must_be_explicit` guards it.

---

## D-010 Exactly one learning-rate controller

**Decision.** `training.lr_schedule` selects one of `reduce_on_plateau`,
`cosine` or `none`.

**Context.** `LearningRateScheduler` *sets* the rate at the start of every
epoch; `ReduceLROnPlateau` *multiplies* the current rate. Run together, the
scheduler overwrites every reduction the plateau callback makes, and the
learning-rate trajectory becomes an artefact of callback ordering rather than a
design.

**Consequences.** The learning-rate trajectory is predictable and explainable.

---

## D-011 Each training stage checkpoints to its own file

**Decision.** `model_stage1.keras` and `model_stage2.keras`, compared at the end;
the genuine winner is copied to `model.keras`.

**Context.** A single shared checkpoint file is subtly wrong across stages: the
stage-2 `ModelCheckpoint` starts with a fresh "best so far", so the first stage-2
epoch always writes, even if it is worse than stage 1's best.

**Consequences.** Fine-tuning can never silently make the shipped model worse,
and the two stages remain individually inspectable. Cost: two extra files per run.

---

## D-012 The predicted class name is derived, never written as a literal

**Decision.** `predicted_class = class_names[int(probability >= threshold)]`, and
the returned `label` field is the same object.

**Context.** With `class_names = ['fake', 'real']` the label index of `real` is
1, so a sigmoid output above the threshold means REAL. Writing class names as
string literals anywhere in the decision path creates two sources of truth for
polarity, and the two can contradict each other, making every reported
prediction ambiguous.

**Consequences.** The class order can change (a different dataset, a different
directory naming) without touching the decision code.
`tests/test_predict.py::test_class_and_label_never_disagree` runs across five
logits and would fail immediately on a regression.

---

## D-013 Classification metrics only; no R-squared

**Decision.** Evaluation reports accuracy against the majority baseline, balanced
accuracy, ROC AUC, PR AUC per class, log loss, Brier score and a calibration
curve. It does not report R-squared, and a test parses the AST of every module to
guarantee no metric can be produced by `np.random.uniform` or `r2_score`.

**Context.** R-squared is a regression statistic measuring explained variance of
a continuous target; for a Bernoulli label the meaningful analogues are Brier
score, log loss and McFadden's pseudo R-squared. Reporting it for a binary
classifier communicates nothing and invites misreading. The AST guard makes the
metric policy enforceable rather than aspirational: every number in
`metrics.json` is a measured classification statistic.

**Consequences.** If a stakeholder genuinely requires an R-squared-shaped number,
the correct answer is McFadden's pseudo R-squared computed from the fitted and
null log likelihoods, and it should be presented alongside AUC, not instead of it.

---

## D-014 Every headline number ships with a baseline and a confidence interval

**Decision.** `metrics.json` reports `majority_baseline` and
`lift_over_baseline` next to accuracy, and a percentile bootstrap 95 % interval
for accuracy and AUC.

**Context.** A bare accuracy figure is uninterpretable. On this dataset the
majority-class baseline is ~53 %, and on a few hundred samples the 95 % interval
spans roughly 5 points either side, so "58 % accuracy" and "marginally above
chance" can describe the same measurement. The baseline and the interval are
what make the number readable.

**Consequences.** The results file cannot overstate the result. Cost: 2,000
bootstrap resamples add a few seconds to evaluation.

---

## D-015 The threshold is chosen on validation and shipped in the contract

**Decision.** `evaluate` selects the operating threshold on the validation split
(default: maximise F1 for the fake class), applies it unchanged to the test
split, and writes it into `label_map.json`.

**Context.** 0.5 is the right threshold only when the classes are balanced and
the error costs are symmetric. Neither holds for forgery detection. And a
threshold tuned on the data you report is a way of making a weak model look
strong.

**Consequences.** The operating point is an explicit, versioned part of the
deployment artifact rather than an implicit default buried in inference code.

---

## D-016 An abstain band, not a bare decision

**Decision.** `evaluation.abstain_margin` defines a band around the threshold in
which the predictor returns `"abstain"` instead of a class, and `metrics.json`
reports how many test images fall in it and the accuracy on the rest.

**Context.** At this model's accuracy, a confident-looking binary verdict about
whether a photograph of a person is fake is actively harmful. Routing uncertain
cases to a human is the only responsible default.

**Consequences.** The system can be tuned toward precision-on-decided-cases by
widening the band, and the trade-off is measured rather than assumed.

---

## D-017 Models are saved with `model.save()`, never pickled

**Decision.** `model.keras` (the Keras v3 zip format). No joblib, no pickle.

**Context.** Pickling a live Keras object binds the artifact to one exact
TensorFlow build, one Python version and one pickle protocol, and unpickling
untrusted files is a security risk. The native format is the supported,
portable serialisation.

**Consequences.** The artifact survives dependency upgrades and can be converted
to SavedModel, ONNX or TFLite. Cost: the file cannot be loaded by anything that
does not have Keras.

---

## D-018 A package with a CLI as the tested implementation

**Decision.** Logic lives in `src/deepfake/*.py`. `cli.py` is the only entry
point.

**Context.** Notebook-only projects accumulate hidden state: cells executed out
of order, functions redefined, results that no linear run can reproduce. A
package makes the code diffable and testable, and an ablation becomes a second
YAML file rather than a copy-pasted cell.

**Consequences.** Restart-and-run-all is `python -m deepfake.cli all`, the
code is under test, and every run is reproducible from its `config.used.yaml`.
Cost: less immediate for interactive exploration, which is what the standalone
notebook (D-022) and the `smoke` command are for.

---

## D-022 The standalone notebook is the primary interface; the package is the second one

**Decision.** `Deepfake Detection.ipynb` at the project root contains every function
it uses and imports nothing from `src/`. The package and CLI remain, tested, as the
interface for ablations and automation. Neither imports the other.

**Context.** A notebook that is only a thin driver over a package is hard to
*read*: understanding any cell means opening a different file, and running it
means setting `PYTHONPATH` and using a terminal. For a project whose primary
audience reads it top to bottom, the notebook has to be the complete artifact.
D-018's reasoning about hidden state and reproducibility stands; the standalone
notebook satisfies it by being committed already executed, in linear order.

**Consequences.** The method now exists in two places and a change to it has to be
made in both, which is a genuine maintenance cost and the reason this is a decision
rather than an accident. In exchange the notebook is truly standalone: it cannot be
broken by a refactor in `src/`, it runs with nothing configured, and it ships
executed so its results are visible without running it. The duplication is bounded
because the notebook is a fixed demonstration rather than the surface that
experiments are run through.

---

## D-021 The baseline configuration is kept because four alternatives failed to beat it

**Decision.** 224 px input, top 40 backbone layers unfrozen, no frequency branch, no
TTA. The alternatives stay available behind config flags but are off by default.

**Context.** Measured, not assumed. Four configurations on the same sealed test split:

| run | accuracy | 95 % CI | ROC AUC |
|---|---|---|---|
| unfreeze_all (154 layers) | 0.6176 | [0.565, 0.670] | 0.6974 |
| baseline (224 px, top 40) | 0.6405 | [0.588, 0.693] | 0.6969 |
| res384 (384 px) | 0.5817 | [0.526, 0.637] | 0.6864 |
| srm (frequency branch) | 0.6373 | [0.585, 0.690] | 0.6743 |

Every confidence interval overlaps every other. Nothing here is established.

Two results are worth recording precisely because they contradict prior
expectations:

- **Higher resolution was predicted to be the biggest win** (CONTEXT.md, v2.0.0
  follow-ups; and the project study PDF). At 384 px it came **last** on accuracy
  while costing about 3x the training time per epoch. The plausible reading is
  that a backbone which is frozen and then only lightly fine-tuned cannot exploit
  the extra detail, and that 1,429 training images cannot support the larger
  effective capacity. The prediction was wrong and is left in place rather than
  edited out.
- **The SRM frequency branch, the most domain-specific idea available, scored
  lowest on AUC.** A second stream trained from scratch on 1,429 images is a lot of
  new parameters for very little data.

**Consequences.** The baseline is kept for simplicity and cost, not because it won.
More importantly, the fact that four changes in a row landed inside the noise band
says the evaluation, not the model, is now the bottleneck: a single 306-image test
split cannot resolve differences of a few points. The next step is k-fold
cross-validation, not another architecture tweak. Anything that claims an
improvement over 0.697 AUC from here should be required to show it survives a fold
split.

---

## D-020 Default threshold strategy is Youden's J, not per-class F1 (supersedes part of D-015)

**Decision.** `evaluation.threshold_strategy` defaults to `youden`. The `f1`
strategy remains available but is no longer the default.

**Context.** Found by running it. With `f1` the first evaluation selected a
threshold of 0.844 and produced fake recall 0.931 against real recall 0.185, for
0.536 accuracy on a 0.529 baseline. Maximising F1 for one class on a
near-balanced problem degenerates toward "predict that class for everything":
recall rises to 1.0 while precision only falls to the class prior, and F1 stays
high the whole way. Switching to Youden's J (maximise `tpr - fpr`) on the same
trained model gave threshold 0.612, balanced recalls of 0.674 and 0.611, and
0.6405 accuracy, a lift of +0.111 over the baseline. The model did not change;
only the operating point did.

**Consequences.** The default operating point is cost-neutral and does not hide a
degenerate classifier behind a healthy-looking F1. When the real error costs are
known and asymmetric, `f1` or `fixed` with an explicit value is the right choice,
and that choice should be justified in this file rather than left as a default.
The episode is also the argument for reporting threshold-free metrics first: ROC
AUC was 0.6969 under both strategies, and only the accuracy moved.

---

## D-019 Redundant dataset copies are reported, not deleted

**Decision.** The pipeline reads one configured root
(`data/real_and_fake_face`). The two other on-disk copies
(`data/real_fake`, `data/real_and_fake_face_detection`) are left alone.

**Context.** The extra copies are redundant, but they are also several hundred
megabytes of the user's data, and deleting them automatically is not a call this
code should make.

**Consequences.** Some wasted disk. Removing them is a one-line manual step
documented in CONTEXT.md, and nothing in the codebase depends on either copy.
