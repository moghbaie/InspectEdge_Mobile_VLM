# InspectEdge: Mobile Visual Inspection AI

**Engineering plan | Version 1.0 | 2026-11-11**  
**Owner:** Staff ML/AI Engineering  
**Training:** Google Colab GPU  
**Deployment:** Android-first, offline-capable; iOS after runtime validation  
**Status:** Design proposal. No training or dataset license audit has been completed.

## 1. Executive decision

Build a phone application that examines a photograph of a **single industrial component**, localizes **visible surface anomalies**, and produces a **grounded, structured inspection report**. Start with one asset family and a small set of defect classes, then expand only after an independent field-image evaluation.

**Recommended MVP domain:** visually inspect industrial parts and surfaces for scratches, cracks, dents, contamination, and missing material, **subject to the actual labels available in selected datasets**. Do not claim electrical safety certification, hidden-fault diagnosis, or general-purpose inspection of arbitrary equipment.

**Architecture decision:** Use a lightweight anomaly detector / object detector as the primary visual evidence generator. Fine-tune a small vision-language model (VLM) for evidence-grounded descriptions, defect taxonomy, uncertainty, and structured reporting. The VLM is not the sole detector or safety authority. The mobile application can fall back to the detector-only pipeline if the VLM is too slow or inaccurate.

**Research hypothesis:** A task-adapted small VLM, conditioned on an image and optionally detector proposals, improves the factual correctness and usefulness of structured inspection reports relative to a pretrained VLM, without materially increasing unsupported defect claims. A separate hypothesis tests whether it improves performance on low-quality photographs.

## 2. Goals, non-goals, and success criteria

### 2.1 User workflow

1. Technician opens the app and chooses an asset category.
2. Camera guidance checks blur, exposure, distance, and framing before capture.
3. User photographs one component, optionally capturing a second angle.
4. On-device detector marks candidate regions and anomaly scores.
5. On-device VLM describes only visible evidence, assigns a supported defect category or `unknown`, and produces schema-valid JSON.
6. UI overlays evidence regions, displays uncertainty, and allows correction.
7. App saves the photo, model versions, predictions, corrections, and exportable inspection record locally. Cloud sync is optional and off by default.

### 2.2 Initial scope

- **Input:** Single RGB smartphone photograph, initially 640–1024 px on the long edge, with original preserved locally. Low-light, compression, motion blur, and partial occlusion included in evaluation.
- **Outputs:** Asset type (limited vocabulary), visible anomaly type, bounding box or region, short evidence description, image-quality status, uncertainty, and `review_required`.
- **Supported outcomes:** `no_visible_anomaly`, `visible_anomaly`, `uncertain`, `unsupported_asset`.
- **No person recognition, face processing, or autonomous safety decisions.**
- **No guarantee that a visually clean image proves the part is safe.**

### 2.3 Proposed acceptance gates (targets, not predictions)

| Gate | Metric | MVP target | Evaluation rule |
|---|---|---:|---|
| Data integrity | Duplicate / split leakage | 0 known cross-split near-duplicate groups | Perceptual hash + embedding/manual review |
| Data integrity | Critical annotation audit agreement | >= 90% | Double-labeled stratified sample; adjudicate disagreements |
| Structured extraction | JSON schema validity | >= 99.5% | Strict parser on locked test set |
| Grounding | Unsupported defect assertion rate | <= 2% | Human adjudication; count unsupported assertions per inspected image |
| Detection | Defect recall | >= 85% on in-domain locked test | Report precision and confidence intervals at fixed operating point |
| Detection | False positive rate | <= 10% on verified normal images | Fixed threshold selected on validation only |
| Robustness | Relative recall drop on realistic degraded test | <= 15% versus clean matched set | Report each corruption family separately |
| Mobile | End-to-end p95 latency | <= 3 s per still image on named target phone | Cold and warm latency reported separately |
| Mobile | Peak app memory | <= 1.5 GB on target phone | Measure real device memory, not model-file size |
| Mobile | Offline completion | 100% of core inspection workflow | Airplane-mode acceptance test |
| Product | Technician correction workflow | 100% editable and traceable | Every model claim links to image/region and model version |

**Gate interpretation:** If a metric is not met, do not hide it in an average. Document the failure, scope reduction, or detector-only fallback. Targets may be revised after a pilot, with written justification before examining the locked test results.

## 3. Data strategy and collection

### 3.1 Dataset candidates and limitations

| Candidate | Why useful | Limitations / required audit |
|---|---|---|
| MVTec AD (official: https://www.mvtec.com/research-teaching/datasets/mvtec-ad) | Industrial anomaly detection; normal/defective images and anomaly masks | Controlled acquisition and narrow asset classes; published research-use restrictions. **Do not assume commercial training rights.** |
| MVTec AD 2 (official project: https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2) | Additional industrial anomaly challenges, including difficult visual conditions | Verify access, labels, license, and benchmark-specific restrictions |
| VisA (Visual Anomaly dataset; search official release from authors) | Diverse industrial anomaly examples | Verify exact source, licensing, categories, annotations, and split protocol before use |
| BTAD (BeanTech Anomaly Detection; verify official release) | Industrial defects with multiple object categories | Verify rights and whether annotations support proposed outputs |
| Internal opt-in phone pilot | Real camera blur, glare, lighting, viewpoints and device variability | Requires collection consent, rights, de-identification, and independent test partition |

**Licensing gate:** Before download/training, record dataset version, publisher, official URL, exact license text, redistribution restrictions, permitted research/commercial use, attribution, and whether derived model weights may be used commercially. A public download is **not** evidence of commercial permission. Keep any restricted research-only model separate from a future commercial release. If license status is unclear, exclude that source from commercial training and obtain legal review.

### 3.2 Data acquisition targets

- Pilot: 2–3 compatible asset categories, with at least 500 labeled images per category if available. Do not synthesize counts to imply real coverage.
- At least 20% of evaluation images should be genuinely captured with phones in field-like conditions when pilot collection permits. This is a **collection target**, not a property of public datasets.
- Cover clean, defective, ambiguous, unsupported, and out-of-distribution inputs.
- Sample across device model, exposure, angle, distance, background, lighting, defect severity, and object size.
- Maintain a dataset manifest with `image_id`, `source_id`, `asset_id`, `capture_session_id`, `device_id` (pseudonymous), `license_id`, `annotation_version`, `split`, `quality_tags`, and checksum.

### 3.3 Annotation specification

Ground truth must distinguish **observed evidence** from **interpretation**:

- `asset_category`: closed vocabulary or `unsupported`.
- `anomaly_present`: yes / no / indeterminate.
- `defect_type`: class from dataset-specific taxonomy or `unknown`.
- `region`: box or mask when annotators can localize it.
- `visible_evidence`: brief literal description (e.g., "thin dark line across the upper surface").
- `image_quality`: blur, noise, underexposure, overexposure, glare, occlusion, resolution, each with ordinal severity.
- `certainty`: sufficient / insufficient visual evidence.
- `review_required`: true if uncertain, severe quality issue, or potentially consequential defect.

Require two annotators on a stratified audit sample. Adjudicate defect-vs-artifact disagreements. Never convert unverified synthetic defects into authoritative labels without provenance.

### 3.4 Leakage-resistant splits

- Group by physical asset, source video/session, capture burst, original image, and derivative/augmentation lineage.
- Keep all versions of one source image in **one** split.
- Suggested initial split: 70% train, 15% validation, 15% locked test **by group**, not by image. Honor official benchmark splits where required and maintain a separate external phone test set.
- Test on at least one held-out device family or acquisition environment when data permits.
- Detect exact duplicates (SHA-256), perceptual near-duplicates (pHash), and visually similar scenes (embedding retrieval + manual audit).
- Do not tune prompts, thresholds, augmentations, or model selection against locked test results.

## 4. Low-quality image augmentation

### 4.1 Training augmentations

Use physically plausible transforms, applied only to training images:

| Family | Examples | Risks to control |
|---|---|---|
| Lighting | Underexposure, uneven illumination, white balance, glare | May conceal true defect or invent contrast |
| Blur | Defocus, directional motion blur, small camera shake | Can erase thin cracks |
| Sensor artifacts | Shot noise, read noise, color noise | May resemble surface texture |
| Compression | JPEG/WebP compression and rescaling | Block artifacts may resemble damage |
| Geometry | Small rotations, perspective, scale, crop | Crop may remove evidence or change label |
| Occlusion | Limited realistic partial obstruction | Do not keep positive label when defect becomes invisible |
| Background | Background variation when asset mask exists | Avoid unnatural composites and shortcut artifacts |

Use mild/moderate/severe corruption schedules. Record random seed, transform parameters, source lineage, and whether defect remains visible. Train with clean and degraded samples. **Do not use aggressive super-resolution to create evidence**. Any enhancement must be compared against the original and treated as potentially misleading.

### 4.2 Label-preservation checks

- For each transform, validate bounding-box/mask geometry.
- Reject crops where a labeled defect is no longer visible.
- Reclassify severe occlusions as `indeterminate` where appropriate.
- Manually inspect 50–100 examples per augmentation family before large-scale training.
- Compare augmented examples to genuine low-quality phone photos to detect unrealistic corruption distributions.

### 4.3 Robustness test matrix

Construct a fixed corruption suite from **held-out originals** with severity 1–5. Report clean and each degradation separately: low light, blur, noise, compression, glare, crop/occlusion, and combined conditions. This suite is for testing, not threshold selection. A separate real-phone set remains the decisive generalization test.

## 5. Models and training plan

### 5.1 Baselines before fine-tuning

**B0: Quality gate + classical/simple baseline.** Reject unusable images; compare a basic anomaly detector or feature-based approach.

**B1: Lightweight industrial anomaly detector.** Evaluate an established anomaly-detection method appropriate to available normal/defect labels (e.g., PatchCore or a lightweight student-teacher alternative), and/or a nano-class object detector if bounding boxes and defect classes exist.

**B2: Pretrained small VLM.** Prompt for the exact output schema without training. Evaluate both full image and detector-proposed crop.

**B3: Detector + rules/template report.** Produce structured reports without a VLM. This is the most important commercial baseline.

### 5.2 VLM candidates

- **First candidate:** SmolVLM2-500M class, selected for mobile-size feasibility; verify that its current checkpoint, tokenizer/processor, image inputs, quantization, and Android runtime are supported before committing. For still images, a compatible image-focused small VLM is also acceptable.
- **Quality challenger:** Qwen2.5-VL-3B-Instruct with 4-bit QLoRA in Colab, **not** assumed deployable at target latency/memory on typical phones.
- **Decision rule:** Prefer the smallest model that clears report-grounding and device gates. A larger VLM that only runs in Colab is a research comparison, not the shipped model.

### 5.3 Supervised training records

Each example contains: image (and optionally one localized crop), explicit task instruction, allowed taxonomy, image-quality tags if measured, and a canonical JSON target. Include hard negatives, ambiguous images, normal samples, unsupported assets, and `unknown` outcomes. Avoid free-form labels not supported by annotations.

Example target (illustrative, not a prediction):

```json
{
  "schema_version": "1.0",
  "asset_category": "metal_part",
  "image_quality": {"usable": true, "issues": ["low_light"]},
  "findings": [
    {
      "type": "scratch",
      "region_xyxy_normalized": [0.24, 0.31, 0.68, 0.44],
      "visible_evidence": "narrow linear surface mark",
      "confidence": 0.72,
      "review_required": true
    }
  ],
  "inspection_status": "visible_anomaly"
}
```

**Important:** Numeric confidence must be calibrated from validation data or reported as a non-probabilistic model score. Do not treat generated confidence text as calibrated probability.

### 5.4 Colab fine-tuning configuration

1. Pin Python, PyTorch, Transformers, PEFT, TRL, Accelerate, and dataset versions; capture `pip freeze`, CUDA version, GPU model, and git commit.
2. Inspect Colab GPU memory at runtime. Free-tier access is variable; **never assume an A100 or fixed session duration**.
3. Train the 500M candidate first with a memory-safe recipe, initially freezing the vision encoder and training language-side/adaptation layers as supported by the model implementation.
4. For the 3B challenger, attempt 4-bit QLoRA with gradient checkpointing, batch size 1, gradient accumulation, conservative image resolution, and short sequences. If it does not fit, reduce resolution or use a larger GPU rather than silently dropping examples.
5. Initial search grid: learning rate 1e-4 / 2e-4 for LoRA adapters, rank 8 / 16, 2–4 epochs, warmup 3–5%, fixed seed plus a second seed for finalists. These are **starting hypotheses**, not guaranteed optimal settings.
6. Use early stopping on validation **grounded-report score** and unsupported-claim rate, not only training loss.
7. Save adapters/checkpoints, processor config, label taxonomy, schema, training manifest, model card, and reproducibility metadata to persistent storage.
8. If full fine-tuning of the 500M model is more stable and fits GPU memory, compare it to LoRA. Do not assume QLoRA is always optimal for the smallest model.

### 5.5 Ablations

- Clean-only vs mixed-quality augmentation.
- Full image vs detector crops vs both.
- Pretrained vs LoRA fine-tuned VLM.
- VLM-only vs detector-only vs detector+VLM.
- With vs without explicit abstention/unknown examples.
- Quantized vs non-quantized inference.

## 6. Evaluation design

### 6.1 Detection metrics

- Per-class precision, recall, F1, AP/mAP where ground-truth regions exist.
- Image-level AUROC and AUPRC for anomaly scores when applicable; report class prevalence.
- False positives per 100 verified-normal images.
- Localization IoU or mask IoU on annotated regions.
- Recall by defect size, lighting, blur, asset category, and device.

### 6.2 VLM/report metrics

- Exact JSON parse success and schema compliance.
- Asset classification macro-F1.
- Defect classification macro-F1, with `unknown` as a legitimate output.
- Evidence grounding: human-rated supported / partially supported / unsupported claims.
- Unsupported defect assertion rate and false reassurance rate (saying "no anomaly" on verified defects).
- Region grounding IoU where coordinates are emitted.
- Selective accuracy vs coverage: performance when model abstains below a validated threshold.
- Report completeness and correction rate in technician review.

**Do not use a single LLM-as-judge score as ground truth.** Use deterministic schema checks, reference labels, and blinded human review of sampled outputs. If an LLM judge is added, measure agreement with humans and audit its errors.

### 6.3 Statistics and error analysis

- Pre-register the primary metric and fixed operating point before locked testing.
- Bootstrap 95% confidence intervals by **asset/session group**, not individual augmented image.
- Compare matched examples between baselines; report absolute differences and uncertainty.
- Create an error taxonomy: missed small defect, glare artifact, texture confusion, crop failure, wrong asset, unsupported claim, calibration failure, schema error.
- Publish confusion matrices and a failure gallery with permitted/redacted examples.
- Repeat finalist training with at least two seeds if compute permits.

### 6.4 Holdout sets

**Validation:** threshold selection, checkpoint selection, prompt revisions.  
**Locked in-domain test:** final quantitative comparison.  
**Real-phone external test:** different camera/device/site and genuine poor lighting.  
**OOD/safety set:** unsupported assets, blank frames, severe blur, nonindustrial scenes, printed photos/screens, and visually ambiguous surfaces.

## 7. Mobile deployment architecture

### 7.1 Android-first pipeline

```text
CameraX capture
  -> image-quality gate (blur/exposure/framing)
  -> detector/anomaly model (mobile runtime)
  -> candidate regions + scores
  -> small quantized VLM (only when needed)
  -> schema validator + calibrated abstention
  -> local report store + technician correction
  -> optional export/sync with explicit permission
```

Candidate runtimes: **LiteRT / TensorFlow Lite, ONNX Runtime Mobile, or a validated vendor acceleration stack** for detector; for VLM, select a runtime only after verifying support for the exact vision encoder, image preprocessing, tokenizer, and quantization format. Model conversion is a separate milestone; a desktop checkpoint is not automatically mobile-compatible.

### 7.2 Mobile constraints and profiling

- Target device must be named in the benchmark (e.g., a midrange Android handset and one flagship). Record SoC, RAM, Android version, NPU/GPU availability, and runtime.
- Quantize detector to INT8 if accuracy passes validation. Evaluate VLM 4-bit or supported alternative on real hardware.
- Benchmark app cold start, warm p50/p95 latency, peak RSS, model load time, battery drain over 50–100 inspections, device temperature, and offline behavior.
- Compare CPU, GPU, and NPU delegates when supported; do not claim acceleration without measurement.
- If the VLM exceeds memory/latency limits, use detector + template reports on device and offer optional user-consented remote interpretation, or reduce the VLM task.
- Prefer still-image analysis for v1. Continuous video inference is out of scope.

### 7.3 Privacy and security

- Default: on-device inference, local encrypted storage where supported, no automatic upload.
- Remove or mask incidental personal information and faces from training and exported examples when possible.
- Explicit consent for pilot image collection and any cloud upload.
- Record model version, timestamp, user edits, and original evidence reference for auditability.
- Protect against prompt injection from visible printed text by treating image text as untrusted evidence, not executable instructions.

## 8. Implementation workstreams and stage gates

| Stage | Work | Deliverables | Exit criteria |
|---|---|---|---|
| 0. Requirements | Choose asset family and defect taxonomy; define target phone and safety limits | PRD, JSON schema, metric contract | Stakeholder approval of measurable scope |
| 1. Dataset audit | Verify official sources, license, counts, labels, rights | `data_sources.csv`, license register | Usable data and permissible research use established |
| 2. Data QA | Download, checksum, inspect, label audit, grouped splits | Manifest, QA report, locked split hashes | No known leakage; quality review passed |
| 3. Augmentation | Implement realistic corruption pipeline and label-preservation tests | Versioned augmentation config, visual QA gallery | Geometry and label checks pass |
| 4. Baselines | Train/evaluate detector and pretrained VLM | Baseline report and failure taxonomy | Primary metrics and runtime recorded |
| 5. VLM SFT | Colab fine-tuning, ablations, reproducible checkpoints | Adapter/weights, train logs, model card | Validation improves without excess unsupported claims |
| 6. Locked evaluation | In-domain, real-phone, OOD, uncertainty analysis | Evaluation report, CI, error gallery | Meets predeclared quality gates or documented fallback |
| 7. Mobile export | Convert, quantize, benchmark on named phones | Android inference module, benchmark traces | p95 latency/memory/accuracy gates met |
| 8. Pilot | Technician study and correction feedback | Prototype APK, usability findings, pilot report | Measurable workflow benefit and no critical unsafe claims |

### Test requirements by stage

- **Unit tests:** annotation schema, bounding-box transforms, corruption determinism, JSON validator, model output parser, confidence thresholding.
- **Data tests:** missing files, label vocabulary, invalid regions, duplicate leakage, license provenance, split integrity.
- **Training tests:** one-batch overfit smoke test, deterministic sample loading, checkpoint resume, NaN/Inf guard, processor-image compatibility.
- **Evaluation tests:** no train/test overlap, locked thresholds, reproducible metrics, grouped bootstrap, explicit empty/unknown handling.
- **Mobile tests:** airplane mode, app background/foreground, low memory, camera permission denial, corrupted model file, poor lighting, interrupted capture, thermal throttling.
- **Security tests:** local data deletion, export permissions, image-text prompt injection, malformed image metadata.

## 9. Suggested repository layout

```text
inspectedge/
  README.md
  PLAN.md
  configs/
    data.yaml
    augmentations.yaml
    train_vlm.yaml
    eval.yaml
  data/
    README.md                  # manifests only; no restricted datasets committed
    licenses/
    manifests/
  notebooks/
    01_dataset_audit_colab.ipynb
    02_baselines_colab.ipynb
    03_vlm_finetune_colab.ipynb
    04_evaluation_colab.ipynb
  src/inspectedge/
    data/
    augmentation/
    detection/
    vlm/
    evaluation/
    schemas/
    export/
  tests/
    unit/
    data/
    integration/
  mobile/android/
  reports/
    dataset_audit.md
    baseline_metrics.md
    finetuning_results.md
    robustness_results.md
    mobile_benchmarks.md
    model_card.md
```

## 10. Experiment tracking and reproducibility

Use MLflow or Weights & Biases if permitted, otherwise local JSON/CSV artifacts synced to persistent storage. Log dataset manifest hash, license snapshot, model and processor revision, seed, GPU type, precision, training time, augmentation config, hyperparameters, validation metrics, checkpoint hash, and evaluation code commit. Never include restricted raw dataset images in public experiment logs.

For every stage, write a concise `stage_report.md` containing objective, inputs, commands, tests run, observed metrics, failure examples, decision, and next-stage approval. A stage is incomplete until its tests and report exist.

## 11. Risk register and mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| Public datasets do not represent phone photos | High | External real-phone evaluation; capture opt-in pilot images |
| Research-only licenses block commercial release | High | License gate; commercial-rights dataset or independently collected data |
| Tiny defects disappear under blur | High | Quality gate, multi-angle capture, abstention, never invent details |
| VLM hallucinates defects | High | Detector grounding, schema constraints, unknown class, human review |
| 3B VLM too slow on phone | High | Ship smaller model or detector-only fallback; benchmark first |
| Defect labels inconsistent across datasets | Medium | Taxonomy mapping, provenance, adjudicated annotation sample |
| Synthetic augmentations fail to match reality | Medium | Real degraded phone holdout and per-corruption reporting |
| Dataset imbalance hides rare-defect failures | High | Per-class metrics, AUPRC, fixed threshold, confidence intervals |
| A visually clean photo is mistaken for safety clearance | High | UI wording: "no visible anomaly detected", never "safe" |

## 12. Eight-week execution proposal

**Week 1:** Define scope, select target phones, audit datasets/licenses, finalize taxonomy and schema.  
**Week 2:** Data ingestion, QA, grouped splits, annotation audit, locked test set.  
**Week 3:** Low-quality augmentation pipeline, visual QA, quality-gate baseline.  
**Week 4:** Detector/anomaly baselines and pretrained VLM benchmark.  
**Week 5:** Colab small-VLM SFT, validation, ablations, checkpoint selection.  
**Week 6:** Locked test, real-phone external evaluation, error analysis, calibration.  
**Week 7:** Mobile conversion, quantization, on-device profiling, fallback decisions.  
**Week 8:** Android prototype, technician feedback, final evaluation and release decision.

Timeline assumes usable datasets, access to a Colab GPU, and a test Android phone. Commercial rights, field-data collection, and mobile runtime compatibility may extend the schedule.

## 13. Immediate first sprint (five working days)

**Day 1:** Select one asset family; define 3–5 observable defect types; create JSON schema and product non-goals.  
**Day 2:** Inspect candidate dataset terms and sample labels; record license decisions; reject mismatched data.  
**Day 3:** Build manifest, checksum, grouping, duplicate detection, and split tests.  
**Day 4:** Run pretrained detector and pretrained VLM on a small stratified sample; collect latency and failure cases.  
**Day 5:** Produce baseline report, decide whether data supports the chosen task, and approve or revise the fine-tuning experiment.

**Stop condition:** If no legally usable dataset contains enough examples of the chosen visible defects, change the initial asset family before training. Do not manufacture a dataset by relabeling unrelated anomalies.

## 14. Release decision

Ship only if the chosen pipeline beats the simplest adequate baseline on a locked external evaluation and meets device limits. If fine-tuning adds no measurable value, ship detector + rules. If mobile VLM inference is too expensive, keep it out of the offline critical path. If false reassurance remains too high, restrict the product to evidence capture and human-assisted reporting.

**Final deliverables:** Reproducible Colab notebooks, data/annotation audit, versioned augmentation pipeline, baseline and fine-tuned models, locked evaluation report with uncertainty, Android prototype, mobile benchmark report, and model card with explicit limitations.
