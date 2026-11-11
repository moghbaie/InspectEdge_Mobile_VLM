# InspectEdge: Execution Plan v2 (Colab, with tests and validation gates)

**Date:** 2026-11-11 · **Builds on:** `InspectEdge_Mobile_VLM_FineTuning_Engineering_Plan.md` (v1.0)
**Runtime:** everything runs in Google Colab (GPU runtime). Code is cloned from GitHub each session; outputs (reports, manifests, checkpoints) persist on Drive at `MyDrive/InspectEdge`. Raw images are downloaded to the Colab local disk each session.
**Status:** data audit is done and the code is smoke-tested locally on a 4 GB GTX 1650. All numbers marked *local smoke* must be re-measured in Colab (notebooks 01–04) before anyone relies on them.

---

## 1. What the v1 plan requires (condensed)

| Requirement | Where it is checked in this plan |
|---|---|
| Photo of a single part → structured JSON report: status, defect type, region, evidence, quality, review flag | Schema + validator `src/inspectedge/schemas/report.py`; Step 5–7 |
| Detector is the primary evidence source; the VLM writes the report and is never the only authority | B1/B3 baselines; release rule in Step 8 |
| Licensed, leakage-free, grouped splits; locked test set | Steps 1–2, `tests/data/test_manifest.py` |
| Realistic low-quality augmentation with label-preservation checks | Step 3, `tests/unit` + visual gallery |
| Baselines B0–B3 before fine-tuning | Step 4 |
| Colab LoRA SFT of a ~500M VLM; 3B challenger is for research only | Steps 5–6 |
| Gates (schema ≥ 99.5%, recall ≥ 85%, FPR ≤ 10%, robustness ≤ 15% drop, unsupported ≤ 2%, …) | Step 7, `scripts/compare_and_gate.py` |
| Mobile export, latency/memory on a named phone | Step 9 (outside Colab; Colab only produces the artifacts) |

---

## 2. Data availability verdict

### 2.1 Sources checked (`data/licenses/data_sources.csv`)

| Source | Obtainable | License | Use in this project |
|---|---|---|---|
| **VisA** (Amazon, ECCV'22): 12 categories, 10,821 images, pixel masks | Yes. Official S3 tar, 1.93 GB, SHA-256 recorded | **CC BY 4.0**: the `LICENSE-DATASET` file inside the archive was verified. Some anomalib docs say CC BY-NC-SA, so this is a known inconsistency for legal review. | **Primary train/val/test** |
| MVTec AD | Yes, via official form or HF mirrors | CC BY-NC-SA 4.0 | Research comparison only; never in the commercial training set |
| MVTec AD 2 | Yes, registration, ~30 GB | CC BY-NC-SA 4.0; test ground truth is held on the server | Optional research robustness benchmark |
| BTAD | Yes, Udine zip | **Unclear** (sources conflict between NC-SA and SA) | Excluded until the license is confirmed |
| MMAD benchmark (ICLR'25): 8,366 images, 39,672 MCQs | Yes, on HF | Research only; images reuse VisA and MVTec | External MLLM benchmark only. **Must be deduplicated against our VisA train split by SHA-256.** |
| Phone photos (field conditions) | **No public source** | n/a | Must be collected (pilot). This is the blocking gap. |

### 2.2 What VisA provides (measured from the manifest)

| | Train | Val | Test |
|---|---:|---:|---:|
| Normal | 6,735 | 1,443 | 1,443 |
| Anomalous | 841 | 180 | 179 |
| PCB subset (pcb1–4), anomalous | 280 | 60 | 60 |

- Every category has exactly 100 anomalous images, about 70 / 15 / 15 per split.
- Defect primary type, mapped into our 9-class taxonomy (`configs/taxonomy_map.yaml`; this draft mapping needs a domain-expert sign-off):

  | Primary type | Train | Val | Test |
  |---|---:|---:|---:|
  | deformation_or_dent | 169 | 42 | 32 |
  | chip_or_missing_material | 146 | 18 | 30 |
  | contamination_or_spot | 140 | 38 | 33 |
  | scratch | 105 | 25 | 21 |
  | missing_or_misplaced_component | 79 | 12 | 22 |
  | discoloration | 77 | 21 | 19 |
  | crack | 68 | 8 | 9 |
  | unknown | 49 | 15 | 9 |
  | foreign_object | 8 | 1 | 4 |
- Defects are **tiny**. Median mask area is 0.19% of the image (p5 = 0.016%, p95 = 2.9%). At a 1024 px long edge, the median defect is about 45×45 px and the p5 defect is about 13×13 px.
- Images come from a fixed rig at ~1.4k×1.1k px with controlled lighting. **No phone photos.**

### 2.3 Is the data enough?

| Purpose | Verdict | Reason |
|---|---|---|
| Detector (B1/B3) training and image-level evaluation | **Enough** | ~560–900 normals per category; 179 test anomalies give roughly ±5 pp CI on aggregate recall |
| VLM SFT for the report format, binary status and localisation | **Enough to train; marginal to prove** | 841 anomalous + about 1.7k sampled normals per epoch. Augmentation multiplies the image-quality variety but **not the defect variety**. |
| Per-defect-type metrics with confidence intervals | **Not enough** | 4–33 test images per type, so CIs are ±15–45 pp. Report per-type numbers as descriptive only and gate on macro-F1 over pooled types. Merge `foreign_object` into `contamination_or_spot` for gating. |
| "Real-phone external test" and "≥ 20% phone images" gates | **Not available** | They cannot be evaluated with public data. Collect **≥ 300 phone images** (≥ 100 with defects, ≥ 2 phone models, ≥ 2 sites) before Step 8. |
| Unsupported-asset / OOD set | **Partly** | Synthesised by asset mismatch (image of category A, user selects B). Still needs real OOD frames: blank, hand, screen, other parts (~200 images, collectable in a day). |

**Decision:** proceed with VisA for Steps 1–7. Use **PCB (pcb1–4) as the MVP asset family**. It is the only "industrial part" family in VisA and it has scratch, bent, melt, missing, dirt and wrong-place defects. Keep the other 8 categories for training diversity and detector robustness, but make no product claims about them. Phone-image collection starts in parallel and gates Step 8.

---

## 3. What was validated and found (local smoke, 2026-11-11)

**Tests:** 70/70 pass (`python -m pytest tests -q`). They cover:
- **Unit:** schema/parser (valid/invalid reports, code fences, braces inside strings), corruption determinism and shapes for all families and severities, mask→box geometry, occlusion visibility, metric correctness (AUROC, AUPRO = 1 on perfect maps, IoU, safety rates, grouped bootstrap).
- **Data:** 10,821 rows; all files exist; SHA-256 spot checks; anomaly ↔ mask ↔ box consistency; valid boxes; closed vocabulary with no unmapped raw labels; official split sanity; no group or exact duplicate crossing splits; split ratios within tolerance; locked split hashes; license text check.

**Findings that changed the design:**
1. **pHash cannot find near-duplicates in rig data.** Different units photographed in the same pose collide at Hamming distance 0 (7,278 such pairs in pcb4 alone). Union-find then chained whole categories into one group, so the "70/15/15" split came out as 7,546 / 2,725 / 550. The fix is to group on 64×64 thumbnail correlation ≥ 0.998 within a category, plus identical SHA-256. That gives 15 pairs, each confirmed by eye as re-shots of the same scene, and a balanced 7,576 / 1,623 / 1,622 split. pHash is still recorded for phone data, where it does work.
2. **Two anomalies had defects below the minimum box area** (candle/056, macaroni2/072) and lost their region. Fixed by falling back to keeping every component. The data test caught this.
3. **Photometric corruption at severity ≥ 4 visibly erases sub-1% defects** (seen in the augmentation gallery), while the mask says "still visible". Such samples are now flagged `defect_visible = None` (unverified) and are never used as positive SFT targets. SFT augmentation is capped at severity 3, and a unit test enforces the cap.
4. **fp16 NaNs on GTX 16-series:** cuDNN returned NaN PatchCore features for batch size 6 in fp16, and those silently became NaN scores and a wrong threshold (first B3 run: recall 0.47). PatchCore now falls back to fp32 on any non-finite output, and the scripts assert finite scores. Colab T4/L4/A100 are not known to be affected, but the guard stays.
5. **The robustness gate in v1 is ill-posed.** See the B1 result below: a clean-trained detector flags *every* degraded image, so recall rises to 1.0 and the "relative recall drop ≤ 15%" gate passes trivially. **New gate:** at the val-frozen clean threshold, require recall drop ≤ 15% **and** FPR increase ≤ 5 pp, reported per corruption family.

**Baselines measured (local smoke, GTX 1650; re-measure in Colab):**

| Baseline | Result |
|---|---|
| B1 PatchCore (WRN50, 224 px, 1% coreset), pcb1 clean | image AUROC **0.961**, AUPRC 0.959, F1-max 0.918, pixel AUROC 0.996, AUPRO@0.3 **0.912**. At the val-chosen threshold: recall 0.957, FPR 0.10. Published PatchCore on VisA reaches ~0.98–0.99 image AUROC at higher resolution, so this is in range. |
| B1 robustness, pcb1, severity 3 | AUROC: low-light 0.917, noise 0.772, glare 0.715, combined 0.646, defocus 0.623, JPEG 0.573, motion 0.554. **FPR at the clean threshold ≈ 100%.** The detector needs degraded normals in its memory bank, or the quality gate has to reject first. |
| B0 quality gate (logistic regression on 8 classical features), val | AUROC 0.881. At 5% clean false-reject it catches defocus 98%, motion 92%, low-light 88%, JPEG 88%, noise 74%, **glare 2%**. Glare needs a dedicated specular-highlight detector. |
| B3 detector + template report, pcb1, VLM-val (166 images, 15 anomalous; memory bank from VLM-train normals) | Schema 100%; recall 0.87; false reassurance 0.13; findings on normals 2.6%; **status_acc_strict 0.964**; region IoU ≥ 0.3 in only 27% (coarse anomaly-map boxes). Threshold chosen on this same val set, so optimistic; the test split is the real number. **This is the bar the fine-tuned VLM must beat.** |
| B2 zero-shot SmolVLM-256M / 500M, 48 val images (2 per category × label), 1024 px | **0/48 valid JSON for both models** (strict accuracy 0.0). Failure modes: prose answers ("The image is a cashew."), echoing the schema template, describing objects that aren't there ("4 slices of yellow onion" for macaroni), and token degeneration. Latency p50/p95 on GTX 1650: 3.7/42 s (256M) and 6.7/61 s (500M), inflated by rambling to the 320-token cap. Peak GPU memory 1.5 / 2.0 GB. |
| SFT pipeline smoke (SmolVLM-256M, LoRA r16, 8 examples, 30 steps, 512 px) | Loss 0.71 → 0.24, **smoke test passed**. Its adapter loads through `eval_vlm.py --adapter`, and JSON parse rate rises from 0% to 75% on 4 pcb1 val images, while the strict validator still rejects degenerate boxes. This verifies the pipeline only; it is not an accuracy result. |

**Consequences:**
- (a) Zero-shot small VLMs are not a usable baseline for the format. Add **B2c: zero-shot + grammar-constrained decoding** (JSON-schema-constrained generation, e.g. `outlines` or `lm-format-enforcer`). This separates "can't follow the format" from "can't see the defect" and tells you how much of the fine-tuning gain is format alone.
- (b) A new metric, **`status_acc_strict`**, counts invalid reports as wrong. The lenient false-reassurance rate scored a perfect 0.0 for models that never answered. Selection and McNemar tests use the strict version.

---

## 4. Evaluation framework: metrics and tools

Use standard protocols so results are comparable to the literature, and keep every metric deterministic or human-adjudicated.

| Layer | Metrics | Protocol / tooling |
|---|---|---|
| Anomaly detector | Image AUROC, AUPRC, F1-max; pixel AUROC; **AUPRO@FPR 0.3**; recall/precision/FPR at a val-frozen threshold; FP per 100 normals | MVTec AD / VisA protocol (Bergmann et al.; Zou et al.). Implemented in `evaluation/metrics.py`. Cross-check one category against **anomalib**'s PatchCore + metrics. |
| Box detector (if a YOLO-nano class detector is added) | mAP@0.5, mAP@0.5:0.95, per-class AP | **pycocotools** / `torchmetrics.detection.MeanAveragePrecision` |
| Report / VLM | JSON parse rate, **schema validity** (strict); status balanced accuracy; anomaly recall; **false-reassurance rate** (says clean on a verified defect); **unsupported-finding rate** on verified normals; defect-type macro-F1 (pooled types); region best-IoU and IoU ≥ 0.3 rate | Deterministic harness `scripts/eval_vlm.py`. Generic VLM harnesses (**VLMEvalKit**, **lmms-eval**) do not cover this task; use them optionally as a general-capability regression check before and after SFT. |
| External MLLM benchmark | MMAD 7 sub-tasks, MCQ accuracy (anomaly discrimination, defect classification/localisation/description/analysis, object classification/analysis) | **MMAD** official scripts (ICLR 2025), after removing items whose image SHA-256 is in our train split. Research-only license. |
| Grounding | Human rating supported / partial / unsupported per claim; **unsupported-assertion rate ≤ 2%**; inter-rater agreement (Cohen's κ ≥ 0.6 before trusting the numbers) | Blinded sheet from `scripts/compare_and_gate.py`, two raters plus adjudication. No LLM judge as ground truth. |
| Calibration / abstention | Risk-coverage curve and AURC; ECE on the detector score after Platt/isotonic calibration on val | Only if `confidence` is emitted; generated confidence text is never treated as a probability. |
| Statistics | 95% CIs by **group bootstrap**; paired **McNemar** test between systems on matched images; second seed for finalists | `metrics.grouped_bootstrap_ci`, `compare_and_gate.py` |
| Mobile (Step 9) | cold/warm p50/p95 latency, peak RSS, battery over 50–100 runs, thermal | Android Studio profiler / `adb shell dumpsys meminfo`; not Colab |

**Pre-registered primary metric:** anomaly recall on the locked test at the val-frozen operating point, with false-reassurance rate as co-primary safety metric. Write it into `reports/preregistration.json` before any test run (notebook 04 enforces this).

---

## 5. Step-by-step plan with tests and validation

Each step has: **Do** (what to run), **Tests** (automated), **Validate** (human or statistical check), **Exit** (gate to the next step). Write a `reports/stage_<n>.md` per step (objective, commands, tests, metrics, failures, decision).

### Step 0 — Scope and contracts (0.5 day, no GPU)
- **Do:** Confirm the MVP asset family (PCB) and the 9-type taxonomy. Freeze schema v1.0. Name a target phone (one mid-range Android plus one flagship).
- **Tests:** `tests/unit` schema tests are green.
- **Validate:** A domain expert signs off `configs/taxonomy_map.yaml`, especially `melt → deformation_or_dent`, `bubble → unknown` and `damage → unknown`.
- **Exit:** PRD, schema and metric contract approved; the gates in §6 accepted.

### Step 1 — Data acquisition and license (Colab 01, ~10 min, CPU is enough)
- **Do:** Run the setup cells. The VisA archive is downloaded and its SHA-256 checked against the recorded value. Review `data_sources.csv`.
- **Tests:** checksum assert in the notebook; `test_license_provenance`.
- **Validate:** Legal review of VisA CC BY 4.0 for commercial use, recording the anomalib inconsistency. MVTec, BTAD and MMAD stay research-only.
- **Exit:** VisA marked APPROVED for research; commercial status recorded.

### Step 2 — Manifest, QA and locked splits (Colab 01, ~10 min)
- **Do:** `python -m inspectedge.data.build_manifest`. The first run writes `split_hashes.json` (commit it to Drive); later runs must reproduce it or fail.
- **Tests:** all of `tests/data` (13 tests: files, checksums, masks↔labels, boxes, vocabulary, groups, duplicates, ratios, lock, license).
- **Validate:** (a) Manually check the 15 merged near-duplicate pairs. (b) Double-label a stratified audit sample of 120 anomalies (10 per category) for defect type and visibility. Require ≥ 90% agreement with the mapped labels (v1 gate) and adjudicate the rest. (c) Look at 20 random normals per category for unlabeled defects.
- **Exit:** 0 cross-split groups; audit agreement ≥ 90%; split hashes locked.

### Step 3 — Augmentation and quality gate (Colab 01, ~15 min)
- **Do:** Galleries for pcb1, pcb4, capsules and candle; run `b0_quality_gate.py`.
- **Tests:** corruption determinism, shape/dtype, mask preservation for photometric transforms, visibility flips under occlusion, severity ≥ 4 marked unverified, SFT cap below the unverified severity.
- **Validate:** Inspect 50–100 augmented samples per family. Check visibility labels on 30 occlusion and 30 severity-3 blur samples per family. Compare against ≥ 30 real low-quality phone photos once available (realism check).
- **Exit:** Gallery approved; quality gate catches ≥ 90% of severity ≥ 3 blur and low-light at ≤ 5% clean rejects. **Glare is currently 2%: add a specular-blob feature, or ship glare as "capture guidance" only.**

### Step 4 — Baselines (Colab 02, T4 ~1.5–2 h)
- **Do:**
  1. B1 PatchCore on all 12 categories with the corruption suite.
  2. B3 on VLM-val: fit memory banks on VLM-train normals, freeze per-category thresholds at 85% recall.
  3. B2 zero-shot SmolVLM-256M and 500M, 10 per (category, label) cell = 240 images; B2c is the same with JSON-schema-constrained decoding.
  4. Optional: Qwen2.5-VL-3B 4-bit on L4/A100.
- **Tests:** B3 template reports must pass the schema validator (asserted per image); the B3 test run refuses to start without frozen thresholds.
- **Validate:** B1 mean image AUROC within ~3 pp of published PatchCore on VisA, otherwise debug. Check one category against anomalib. Error taxonomy for 30 B2 failures (schema error, hallucinated finding, missed defect, wrong type).
- **Exit:** baseline table in `reports/comparison_val.md`; decision recorded on whether the VLM gets detector crops as input (ablation in Step 6).

### Step 5 — SFT data and training smoke test (Colab 03, ~10 min)
- **Do:** `train_vlm_lora.py --smoke` overfits 8 examples for 30 steps.
- **Tests:** loss must fall below 50% of its start (asserted); NaN/Inf guard; the collator masks prompt and image tokens (label = −100); deterministic epoch plan (seeded); processor/image compatibility (smoke completes); checkpoint save and `--resume` (interrupt once on purpose and resume).
- **Validate:** Decode 5 training targets and check that each is canonical JSON that passes `validate_report`. Look at 10 augmented SFT samples together with their targets (issues match the augmentation; hidden defects give `uncertain`; mismatch items give `unsupported_asset`).
- **Exit:** smoke passes; resume works.

### Step 6 — Fine-tuning runs and ablations (Colab 03, T4 ~1–1.5 h per run; L4/A100 recommended for the grid)
- **Do:** Main run: SmolVLM-500M, LoRA r16, lr 2e-4, 3 epochs, longest edge 1024, vision encoder frozen, aug_prob 0.4, 2:1 normals, 5% asset mismatch. Grid: {1e-4, 2e-4} × {r8, r16}. Ablations: clean-only, no-abstention, PCB-only, full image vs detector crop, second seed for the top 2.
- **Tests:** NaN guard; `run_meta.json` holds the pip freeze, GPU, split hashes and config.
- **Validate:** Evaluate on **val only** (notebook 04 §1). Early-stop and select on val `status_acc_strict`, subject to schema ≥ 99.5% and unsupported findings on normals ≤ 10%, not on training loss.
- **Exit:** at least one run beats **both** B2/B2c and B3 on val `status_acc_strict`, with no worse false-reassurance (McNemar p < 0.05 vs B2). If none beats B3, record that and go to the detector-only path.

### Step 7 — Locked evaluation (Colab 04, ~1 h)
- **Do:** Write `preregistration.json`, then run B3, B2 and FT on the **test split once**. Run the robustness suite (7 families at severity 3) and `compare_and_gate.py`. Two raters fill in the human-review sheet. Optionally run MMAD (deduplicated).
- **Tests:** the notebook asserts that pre-registration exists; frozen thresholds are reused; no image in test shares a group with train (data test).
- **Validate:** grouped bootstrap CIs; paired McNemar; error gallery; κ ≥ 0.6 between raters.
- **Exit:** gates in §6 met, or failures documented with a fallback decision.

### Step 8 — Real-phone external test (Colab 04 with a phone manifest; needs collection)
- **Do:** Collect ≥ 300 consented phone photos (PCB family, ≥ 2 devices, ≥ 2 sites, including genuinely poor lighting) and ~200 OOD frames. Annotate with the Step 2 protocol, add them to the manifest as `source_id=internal_phone_pilot` with `split=external_test`, and re-run the Step 7 evaluation unchanged.
- **Validate:** compare in-domain vs phone recall and FPR per device.
- **Exit:** phone recall ≥ 85% × 0.85 (≤ 15% relative drop) with FPR ≤ 15%; otherwise collect phone training data and repeat Steps 6–7.

### Step 9 — Mobile export (outside Colab, except the conversion)
- Colab produces: PatchCore → ONNX → INT8 (LiteRT or ONNX Runtime Mobile); merged LoRA → ONNX or a llama.cpp/GGUF-class 4-bit build, depending on the runtime chosen after verifying the SmolVLM vision path. Re-run the val harness on the **quantized** models in Colab (accuracy must stay within 2 pp) before any device benchmark.
- The device benchmark (p95 ≤ 3 s, RSS ≤ 1.5 GB, airplane mode) happens on the named phone.

---

## 6. Acceptance gates (v2)

| Gate | v1 | v2 change |
|---|---|---|
| Split leakage | 0 near-duplicate groups across splits | Grouping by thumbnail correlation (pHash is invalid for rig data); test enforced |
| Annotation audit | ≥ 90% | Unchanged; 120-image stratified sample |
| Schema validity | ≥ 99.5% | Unchanged; strict validator plus cross-field rules (x1<x2, status ↔ findings); split lock enforced on every rebuild |
| Recall | ≥ 85% locked test | At the **val-frozen** threshold; CI reported; PCB subset reported separately |
| False positives | ≤ 10% on normals | Unchanged; also report FP per 100 normals |
| **False reassurance** | (metric only) | **New gate: ≤ 15% on test, ≤ B3** |
| Robustness | ≤ 15% relative recall drop | **And FPR increase ≤ 5 pp at the clean threshold**, per family; the quality gate may reject first (then report the rejection rate) |
| Unsupported assertions | ≤ 2% | Human-adjudicated, κ ≥ 0.6; the automated proxy (findings on normals) is a screen only |
| Per-type metrics | per-class P/R/F1 | Descriptive only (n = 4–33); gate on pooled macro-F1 ≥ B3 + 0.10 (B3 cannot name types) |
| Phone external test | ≥ 20% of eval | **Blocking for release**; needs collection (Step 8) |

---

## 7. Colab budget (T4 unless noted)

| Notebook | Wall time | Notes |
|---|---|---|
| 01 audit | ~25 min | Download ~1–2 min; manifest ~5 min; tests ~1 min |
| 02 baselines | ~2 h | PatchCore 12 categories + corruptions ~1 h; B3 ~40 min; B2 2×240 images ~20 min |
| 03 fine-tune | ~1–1.5 h per run | Main run plus 3 grid runs plus ~5 ablations ≈ 10–12 GPU-h → spread across sessions or use L4/A100 |
| 04 evaluation | ~1.5 h per finalist | Full val 1,623 images plus test 1,622 images at ~1 s per image, plus robustness |

Colab sessions disconnect. All long steps are resumable (`--resume`), checkpoints go straight to Drive, and every notebook ends with an rsync of `reports/` and `data/manifests/`.

## 8. Open decisions and risks
1. **Phone data** is the single largest gap. Without it, only the "in-domain rig" claims can be made.
2. The VisA license inconsistency needs legal confirmation before any commercial model.
3. Glare is undetected by the classical gate.
4. A clean-trained detector collapses on degraded photos. Choose between (a) adding mildly degraded normals to the memory bank, and (b) gate-then-detect. Ablate both in Step 6.
5. The taxonomy mapping is a draft that needs expert review.
6. SmolVLM's mobile runtime path (vision encoder + image splitting) must be verified before Step 9. If it fails, ship B3 (detector + template) and keep the VLM optional.
