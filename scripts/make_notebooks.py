"""Generate the Colab notebooks in notebooks/ (single source; re-run after editing)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VISA_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
VISA_SHA = "2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}


SETUP = [
    md("""
## 0. Session setup (run every session)
Code is cloned from GitHub (`moghbaie/InspectEdge_Mobile_VLM`); nothing needs to be uploaded.
Google Drive (`MyDrive/InspectEdge`) only stores outputs: reports, manifests, checkpoints.
Runtime → Change runtime type → **GPU** (T4 is enough for 500M; L4/A100 for 3B).
"""),
    code("""
import os, json, subprocess
from google.colab import drive
drive.mount('/content/drive')
REPO  = 'https://github.com/moghbaie/InspectEdge_Mobile_VLM.git'
DRIVE = '/content/drive/MyDrive/InspectEdge'   # persistent outputs: reports, manifests, checkpoints
WORK  = '/content/InspectEdge'                 # fast local disk: code + raw images
os.makedirs(DRIVE, exist_ok=True)
if os.path.isdir(f'{WORK}/.git'):
    !git -C {WORK} pull -q --ff-only
else:
    !git clone -q {REPO} {WORK}
assert os.path.exists(f'{WORK}/pyproject.toml'), 'clone failed: check REPO URL / repo visibility'
%cd {WORK}
!git log --oneline -1
!pip -q install -r requirements-colab.txt
!pip -q install -e . --no-deps
os.environ['INSPECTEDGE_ROOT'] = WORK
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
import torch, transformers, peft
assert transformers.__version__ == '4.57.1', 'pinned transformers not active: Runtime -> Restart session, then re-run this cell'
print('torch', torch.__version__, 'cuda', torch.version.cuda, '| transformers', transformers.__version__, '| peft', peft.__version__)
assert torch.cuda.is_available(), 'No GPU: Runtime -> Change runtime type -> GPU'
"""),
    code(f"""
# Data: official VisA release (CC BY 4.0). Verified by SHA-256 before use.
VISA_SHA = '{VISA_SHA}'
import pandas as pd
def visa_missing():   # every manifest image/mask must be on local disk (a runtime reset or partial extract loses them)
    man = pd.read_json('data/manifests/visa_manifest.jsonl', lines=True)
    return [p for p in list(man.image_path) + [m for m in man.mask_path if m] if not os.path.exists(p)]
if visa_missing():
    !rm -rf data/raw/visa data/raw/_dl
    os.makedirs('data/raw/_dl', exist_ok=True); os.makedirs('data/raw/visa', exist_ok=True)
    !wget -q -O data/raw/_dl/VisA_20220922.tar {VISA_URL}
    h = subprocess.run(['sha256sum', 'data/raw/_dl/VisA_20220922.tar'], capture_output=True, text=True).stdout.split()[0]
    assert h == VISA_SHA, f'VisA archive checksum mismatch: {{h}}'
    !tar -xf data/raw/_dl/VisA_20220922.tar -C data/raw/visa && rm data/raw/_dl/VisA_20220922.tar
_miss = visa_missing()
assert not _miss, f'{{len(_miss)}} VisA files missing after extraction (disk full?), e.g. {{_miss[0]}}'
open('data/raw/visa/.complete', 'w').close()
!ls data/raw/visa
"""),
    code("""
# Manifest: rebuilt deterministically each session; fails loudly if the locked split changed.
!python -m inspectedge.data.build_manifest
"""),
]

SYNC = code("""
# Persist outputs to Drive (run at the end and after long steps)
!rsync -aR reports data/manifests {DRIVE}/
print('synced to', DRIVE)
""")


GPU_CHECK = code("""
# Ampere+ GPU (A100 preferred; L4/H100 also fine). bf16 + TF32 need compute capability >= 8.0.
gpu = torch.cuda.get_device_name(0); mem_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
assert torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability(0)[0] >= 8, \\
    f'{gpu} lacks bf16: Runtime -> Change runtime type -> A100 (or L4) GPU'
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
print(gpu, f'{mem_gb:.0f} GB')
""")
A100_NBS = ("03_", "04_", "05_")

# Colab wipes /content (images, local reports/) and the Python state on a runtime reset.
GUARD = ("import os\nassert os.path.exists('/content/InspectEdge/data/raw/visa/.complete') and os.getcwd() == '/content/InspectEdge', "
         "'Runtime was reset: re-run every cell from section 0 (setup) first'\n")


def nb(cells, gpu="T4"):
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": gpu, "provenance": []},
                                          "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                          "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


NB = {
    "01_dataset_audit_colab.ipynb": [
        md("# 01 · Dataset audit, validation and augmentation QA\nExit criteria: all unit + data tests pass, split lock verified, license register reviewed, augmentation gallery approved by a human."),
        *SETUP,
        md("## 1. Unit + data validation tests"),
        code("!python -m pytest tests -q -rA 2>&1 | tail -25"),
        md("## 2. License register (manual gate)"),
        code("import pandas as pd; pd.read_csv('data/licenses/data_sources.csv')[['source_id','license','commercial_training','status']]"),
        md("## 3. Data sufficiency statistics"),
        code("!python scripts/dataset_stats.py"),
        md("## 4. Augmentation visual QA (approve 50–100 examples per family before training)"),
        code("""
from IPython.display import Image, display
for c in ['pcb1', 'pcb4', 'capsules', 'candle']:
    !python scripts/aug_gallery.py {c}
    display(Image(f'reports/aug_gallery_{c}.jpg', width=700))
"""),
        md("## 5. B0 image-quality gate"),
        code("!python scripts/b0_quality_gate.py"),
        SYNC,
    ],
    "02_baselines_colab.ipynb": [
        md("# 02 · Baselines B1 (PatchCore), B3 (detector + template report), B2 (zero-shot VLMs)\nAll thresholds are chosen on **val**. Do not run `--split test` here."),
        *SETUP,
        md("## B1 · PatchCore on the official VisA 1-class split (+ corruption suite, severity 3)\n~3–5 min/category on T4."),
        code("!python scripts/baseline_patchcore.py --corrupt 2>&1 | grep -v Warning"),
        code("""
r = json.load(open('reports/baseline_patchcore/metrics.json'))
import pandas as pd
cats = [c for c in r if not c.startswith('_')]
pd.DataFrame({c: {k: r[c][k] for k in ['image_auroc','image_auprc','image_f1_max','pixel_auroc','aupro_0.3']} for c in cats}).T.round(3)
"""),
        code("""
rob = pd.DataFrame({c: {f: v['image_auroc'] for f, v in r[c]['robustness_sev3'].items()} for c in cats}).T
rob.insert(0, 'clean', [r[c]['image_auroc'] for c in cats]); rob.round(3)
"""),
        md("## B3 · Detector + template report on the VLM val split (freezes per-category thresholds)"),
        code("!python scripts/b3_detector_report.py --split val 2>&1 | grep -v Warning | tail -25"),
        md("## B2 · Zero-shot small VLMs (same prompt/schema as fine-tuning)"),
        code("""
for m in ['HuggingFaceTB/SmolVLM-256M-Instruct', 'HuggingFaceTB/SmolVLM-500M-Instruct']:
    !python scripts/eval_vlm.py --model {m} --split val --per-cell 10 2>&1 | grep -v Warning | tail -30
"""),
        md("Optional quality challenger (needs L4/A100, 4-bit; research comparison only — not deployable on phone):"),
        code("# !python scripts/eval_vlm.py --model Qwen/Qwen2.5-VL-3B-Instruct --load-4bit --split val --per-cell 5"),
        code("""
!python scripts/compare_and_gate.py --split val --system B3=reports/b3_detector_report/val_predictions.jsonl --system B2_500M=reports/vlm_eval/SmolVLM-500M-Instruct_val_predictions.jsonl
"""),
        SYNC,
    ],
    "03_vlm_finetune_colab.ipynb": [
        md("# 03 · LoRA SFT of SmolVLM-500M (A100)\nOrder: smoke test → single run → small grid. Checkpoints are written straight to Drive so a disconnect only loses minutes (re-run with `--resume`).\nRuntime → Change runtime type → **A100 GPU** (L4 also works; batch settings adapt to GPU memory)."),
        *SETUP,
        GPU_CHECK,
        code("""
# Memory-sized overrides. Effective batch stays 16 (= T4 config 4x4) so lr/warmup in configs/train_vlm.yaml remain valid.
if mem_gb > 60:   BS, CKPT_ = 16, False   # A100/H100 80GB
elif mem_gb > 30: BS, CKPT_ = 8, False    # A100 40GB
else:             BS, CKPT_ = 8, True     # L4 24GB
GPU_SET = f'batch_size={BS} grad_accum={16 // BS} grad_ckpt={str(CKPT_).lower()} workers={min(8, os.cpu_count())} tf32=true'
print(GPU_SET)
"""),
        md("## 1. Training tests: one-batch overfit smoke test (must print SMOKE TEST PASSED)"),
        code("!python scripts/train_vlm_lora.py --smoke --set output_dir={DRIVE}/checkpoints {GPU_SET} 2>&1 | grep -vE 'Warning|warn' | tail -15"),
        md("## 2. Main run (bf16; on OOM append `grad_ckpt=true`, or halve `batch_size` and double `grad_accum`)"),
        code("!python scripts/train_vlm_lora.py --resume --set output_dir={DRIVE}/checkpoints {GPU_SET} 2>&1 | grep -vE 'Warning|warn' | tail -30"),
        code("!nvidia-smi --query-gpu=memory.used,memory.total --format=csv"),
        code("""
meta = json.load(open(f'{DRIVE}/checkpoints/smolvlm500m_lora_r16_lr2e-4_s0/run_meta.json'))
import matplotlib.pyplot as plt; plt.plot(meta['losses']); plt.title('train loss'); plt.show()
print(meta['gpu'], round(meta['train_seconds']/60,1), 'min')
"""),
        md("## 3. Grid (plan 5.4) — run across sessions if needed; each run resumable"),
        code("""
GRID = [(1e-4, 8), (1e-4, 16), (2e-4, 8)]   # (2e-4,16) is the main run above
for lr, r in GRID:
    name = f'smolvlm500m_lora_r{r}_lr{lr:g}_s0'
    !python scripts/train_vlm_lora.py --resume --set output_dir={DRIVE}/checkpoints {GPU_SET} run_name={name} lr={lr} lora_r={r} lora_alpha={2*r} 2>&1 | tail -3
"""),
        md("## 4. Ablations (plan 5.5) — same eval harness in notebook 04; always keep `{GPU_SET}` in `--set`\n- clean-only: `--set aug_prob=0.0 run_name=..._cleanonly`\n- no abstention/mismatch: `--set mismatch_frac=0.0`\n- PCB-only MVP: `--set categories=[pcb1,pcb2,pcb3,pcb4]`\n- second seed for finalists: `--set seed=1 run_name=..._s1`"),
        SYNC,
    ],
    "04_evaluation_colab.ipynb": [
        md("# 04 · Validation selection → locked test → robustness → gates (A100)\n**Rule:** choose checkpoint + thresholds on val only. Write the chosen run name into `reports/preregistration.json` *before* any `--split test` call.\nRuntime → Change runtime type → **A100 GPU** (L4 also works). Note: `latency_p95_s` measured here is server-GPU latency, not on-device; compare systems only when run on the same GPU type."),
        *SETUP,
        GPU_CHECK,
        code("""
CKPT = f'{DRIVE}/checkpoints'
!ls {CKPT}
# Restore earlier outputs (B3 frozen val thresholds from notebook 02, val metrics) that a runtime reset wiped from local disk.
!mkdir -p reports && rsync -a --update {DRIVE}/reports/ reports/ 2>/dev/null; ls reports
"""),
        md("## 1. Validation selection (full val split)"),
        code(GUARD + """
RUNS = ['smolvlm500m_lora_r16_lr2e-4_s0']   # add grid runs
for run in RUNS:
    assert os.path.isdir(f'{CKPT}/{run}/final'), f'no adapter at {CKPT}/{run}/final'
    !python scripts/eval_vlm.py --model HuggingFaceTB/SmolVLM-500M-Instruct --adapter {CKPT}/{run}/final --split val --per-cell 0 --tag {run} 2>&1 | tail -22
    assert os.path.exists(f'reports/vlm_eval/{run}_val_metrics.json'), f'val eval of {run} failed: see the log above'
!rsync -aR reports {DRIVE}/
"""),
        code("""
import glob, pandas as pd
KEYS = ['schema_valid_rate','status_acc_strict','anomaly_balanced_acc','anomaly_recall_flagged','false_reassurance_rate',
        'unsupported_finding_rate_on_normals','defect_type_macro_f1','region_iou>=0.3_rate','latency_p95_s']
rows = {}
for f in sorted(glob.glob(f'{DRIVE}/reports/vlm_eval/*_val_metrics.json')) + sorted(glob.glob('reports/vlm_eval/*_val_metrics.json')):
    m = json.load(open(f))
    if m.get('corrupt'): continue
    rows[os.path.basename(f).replace('_val_metrics.json', '')] = {k: m.get(k) for k in KEYS}   # local (newer) wins
assert rows, 'no val metrics found locally or on Drive: run the cell above first'
pd.DataFrame(rows).T.sort_values('status_acc_strict', ascending=False).round(3)
"""),
        md("Selection rule (pre-registered): among runs with schema_valid ≥ 0.995 and unsupported_finding_rate_on_normals ≤ 0.10 on val, pick max status_acc_strict (invalid report = wrong); ties → lower false_reassurance."),
        code("""
FINAL = 'smolvlm500m_lora_r16_lr2e-4_s0'   # <- set from the table above
prereg = {'final_run': FINAL, 'primary_metric': 'anomaly_recall_flagged @ val-frozen operating point',
          'gates': 'plan 2.3 + FPR-aware robustness gate', 'b3_thresholds': 'reports/b3_detector_report/frozen_thresholds.json'}
os.makedirs('reports', exist_ok=True); json.dump(prereg, open('reports/preregistration.json','w'), indent=2)
!rsync -aR reports {DRIVE}/
"""),
        md("## 2. Locked test — run ONCE"),
        code(GUARD + """
assert os.path.exists('reports/preregistration.json'), 'run the pre-registration cell above first'
# B3 test reuses thresholds frozen on val (normally by notebook 02); freeze them here if they were never synced to Drive.
if not os.path.exists('reports/b3_detector_report/frozen_thresholds.json'):
    !python scripts/b3_detector_report.py --split val 2>&1 | tail -5
!python scripts/b3_detector_report.py --split test 2>&1 | tail -5
!python scripts/eval_vlm.py --model HuggingFaceTB/SmolVLM-500M-Instruct --split test --per-cell 0 2>&1 | tail -5
!python scripts/eval_vlm.py --model HuggingFaceTB/SmolVLM-500M-Instruct --adapter {CKPT}/{FINAL}/final --split test --per-cell 0 --tag {FINAL} 2>&1 | tail -5
SYSTEMS = {'B3': 'reports/b3_detector_report/test_predictions.jsonl',
           'B2': 'reports/vlm_eval/SmolVLM-500M-Instruct_test_metrics.json',
           'FT': f'reports/vlm_eval/{FINAL}_test_metrics.json'}
failed = [k for k, p in SYSTEMS.items() if not os.path.exists(p)]
assert not failed, f'test run failed for {failed}: see the logs above'
!rsync -aR reports {DRIVE}/
!python scripts/compare_and_gate.py --split test --system B3=reports/b3_detector_report/test_predictions.jsonl --system B2=reports/vlm_eval/SmolVLM-500M-Instruct_test_predictions.jsonl --system FT=reports/vlm_eval/{FINAL}_test_predictions.jsonl
!rsync -aR reports {DRIVE}/
"""),
        md("## 3. Robustness suite on held-out test originals (report each family separately)"),
        code(GUARD + """
for fam in ['low_light','defocus_blur','motion_blur','sensor_noise','jpeg','glare','combined']:
    !python scripts/eval_vlm.py --model HuggingFaceTB/SmolVLM-500M-Instruct --adapter {CKPT}/{FINAL}/final --split test --per-cell 5 --corrupt {fam} --severity 3 --tag {FINAL} 2>&1 | tail -22
    assert os.path.exists(f'reports/vlm_eval/{FINAL}_test_{fam}3_metrics.json'), f'robustness eval {fam} failed: see the log above'
!rsync -aR reports {DRIVE}/
"""),
        md("## 4. Human grounding review\nDownload `reports/human_review_test.csv`, have two raters fill it blind, adjudicate disagreements, then compute the unsupported-assertion rate (gate ≤ 2%)."),
        SYNC,
    ],
    "05_gemma3n_finetune_colab.ipynb": [
        md("# 05 · Gemma 3n E2B challenger: LoRA SFT + val comparison (A100)\nSame data, prompt, schema, augmentation and optimiser settings as the SmolVLM-500M runs (`configs/train_gemma3n_e2b.yaml`), so differences come from the base model.\n**Val only.** This notebook never touches `--split test`. If Gemma wins on val, update `reports/preregistration.json` (notebook 04) *before* the locked test.\nRuntime → Change runtime type → **A100 GPU** (L4 works with smaller batches + gradient checkpointing)."),
        *SETUP,
        GPU_CHECK,
        md("## Hugging Face access (Gemma is gated)\n1. Accept the Gemma terms at https://huggingface.co/google/gemma-3n-E2B-it.\n2. Add a read token as a Colab secret named `HF_TOKEN` (🔑 icon in the left sidebar) and enable notebook access."),
        code("""
from google.colab import userdata
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
!pip -q install "timm>=1.0.16"   # Gemma 3n vision tower (MobileNet-V5) is a timm model
from huggingface_hub import hf_hub_download
MODEL = 'google/gemma-3n-E2B-it'
hf_hub_download(MODEL, 'config.json')   # fails here if the license is not accepted / token missing
print('access OK:', MODEL)
"""),
        code("""
# Memory-sized overrides; effective batch stays 16 so lr/warmup match the SmolVLM runs.
# Gemma 3n is ~5B raw params with a 262k vocab (large logits), so batches are smaller than SmolVLM's.
if mem_gb > 60:   BS, CKPT_ = 8, False   # A100/H100 80GB
elif mem_gb > 30: BS, CKPT_ = 4, True    # A100 40GB
else:             BS, CKPT_ = 2, True    # L4 24GB
GPU_SET = f'batch_size={BS} grad_accum={16 // BS} grad_ckpt={str(CKPT_).lower()} workers={min(8, os.cpu_count())} tf32=true'
CFG = 'configs/train_gemma3n_e2b.yaml'
RUN = 'gemma3n_e2b_lora_r16_lr2e-4_s0'
CKPT = f'{DRIVE}/checkpoints'
print(GPU_SET)
"""),
        md("## 1. Zero-shot B2 reference (same 10-per-cell val sample as the SmolVLM B2 baseline)"),
        code("!python scripts/eval_vlm.py --model {MODEL} --split val --per-cell 10 2>&1 | grep -vE 'Warning|warn' | tail -25"),
        md("## 2. One-batch overfit smoke test (must print SMOKE TEST PASSED)\nWritten to `checkpoints/smoke` (shared with notebook 03; it is only a pass/fail check)."),
        code("!python scripts/train_vlm_lora.py --config {CFG} --smoke --set output_dir={CKPT} {GPU_SET} 2>&1 | grep -vE 'Warning|warn' | tail -15"),
        md("## 3. Main run (resumable: re-run this cell after a disconnect; on OOM halve `batch_size`, double `grad_accum`, set `grad_ckpt=true`)"),
        code("!python scripts/train_vlm_lora.py --config {CFG} --resume --set output_dir={CKPT} {GPU_SET} 2>&1 | grep -vE 'Warning|warn' | tail -30"),
        code("!nvidia-smi --query-gpu=memory.used,memory.total --format=csv"),
        code("""
meta = json.load(open(f'{CKPT}/{RUN}/run_meta.json'))
import matplotlib.pyplot as plt; plt.plot(meta['losses']); plt.title(f'{RUN} train loss'); plt.show()
print(meta['gpu'], round(meta['train_seconds']/60,1), 'min')
"""),
        md("""
## 4. Full-val evaluation of the fine-tuned adapter (resumable)
Each prediction is appended to `{DRIVE}/reports/vlm_eval/<tag>_val_predictions.jsonl` and fsynced, so a crash or disconnect loses at most one batch.
After a disconnect: re-run section 0 + the GPU / HF / settings cells (5, 7, 8), then this cell; it continues where it stopped.
`EVAL_CHUNK` images are predicted per call; the cell loops over chunks (each a fresh process, so leaked memory is freed) and writes `<tag>_val_metrics.json` once every val image is done.
Generation is batched (`EVAL_BS`); latency p50/p95 is then re-measured one image at a time on 30 val images, so it stays comparable with the batch-1 SmolVLM rows (`throughput_img_per_s` is the batched rate).
"""),
        code("""
EVAL_DIR = f'{DRIVE}/reports/vlm_eval'   # on Drive: survives runtime resets
EVAL_CHUNK = 300                           # images per subprocess; lower it if a chunk still dies
EVAL_BS = 16 if mem_gb > 60 else 8 if mem_gb > 30 else 4   # batched generate; auto-halves on OOM

def resumable_eval(tag, model_args, bs=EVAL_BS):
    pred, metrics = f'{EVAL_DIR}/{tag}_val_predictions.jsonl', f'{EVAL_DIR}/{tag}_val_metrics.json'
    n_done = lambda: sum(1 for _ in open(pred)) if os.path.exists(pred) else 0
    stalls = 0
    while stalls < 2:   # stop if two chunks in a row add no predictions (a real error: read the log above)
        if os.path.exists(metrics) and os.path.exists(pred) and os.path.getmtime(metrics) >= os.path.getmtime(pred):
            break
        before = n_done()
        !python scripts/eval_vlm.py {model_args} --split val --per-cell 0 --tag {tag} \\
            --resume --out-dir {EVAL_DIR} --max-items {EVAL_CHUNK} --batch-size {bs} 2>&1 | grep --line-buffered -vE 'Warning|warn' | tail -25
        stalls = stalls + 1 if n_done() == before else 0
    assert os.path.exists(metrics), f'{tag}: two chunks in a row added no predictions -- eval_vlm.py is crashing, see the traceback above (re-running will not help)'
    !mkdir -p reports/vlm_eval && cp {metrics} {pred} reports/vlm_eval/   # local copy for section 5
    m = json.load(open(metrics))
    print(json.dumps({k: v for k, v in m.items() if k != 'per_category'}, indent=2))
    return m

m_bf16 = resumable_eval(RUN, f'--model {MODEL} --adapter {CKPT}/{RUN}/final')
"""),
        md("""
## 4b. 4-bit accuracy check: merge the LoRA, quantize to NF4, re-run full val
Execution plan Step 9: the quantized model must stay within **2 pp** of the bf16 model on val before any device work.
The adapter is merged into the base weights first (`{CKPT}/<run>/merged`, ~11 GB bf16 on Drive, written once), then loaded in 4-bit NF4 (bitsandbytes).
This measures the accuracy cost of 4-bit weights; it is not the phone format (LiteRT-LM / GGUF conversion is chosen in Step 9) and it is not faster on an A100.
"""),
        code("""
MERGED = f'{CKPT}/{RUN}/merged'
!python scripts/merge_lora.py --model {MODEL} --adapter {CKPT}/{RUN}/final --out {MERGED} 2>&1 | grep -vE 'Warning|warn' | tail -5
m_nf4 = resumable_eval(f'{RUN}_merged_nf4', f'--model {MERGED} --load-4bit')

ACC = ['schema_valid_rate', 'status_acc_strict', 'anomaly_balanced_acc', 'anomaly_recall_flagged', 'false_reassurance_rate',
       'unsupported_finding_rate_on_normals', 'defect_type_macro_f1', 'region_iou>=0.3_rate']
import pandas as pd
q = pd.DataFrame({'bf16 + adapter': {k: m_bf16.get(k) for k in ACC + ['peak_gpu_mem_gb', 'throughput_img_per_s']},
                  'merged NF4':     {k: m_nf4.get(k)  for k in ACC + ['peak_gpu_mem_gb', 'throughput_img_per_s']}})
q['delta_pp'] = (q['merged NF4'] - q['bf16 + adapter']) * 100
q.loc[['peak_gpu_mem_gb', 'throughput_img_per_s'], 'delta_pp'] = None
worst = q.loc[ACC, 'delta_pp'].abs().max()
print(f'largest accuracy change: {worst:.1f} pp ->', 'PASS (<= 2 pp)' if worst <= 2 else 'FAIL (> 2 pp): try 8-bit or keep bf16 weights for the sensitive layers')
q.round(3)
"""),
        md("## 5. Val comparison with the SmolVLM runs (metrics from this session + those synced to Drive by notebook 04)"),
        code("""
import glob, pandas as pd
KEYS = ['schema_valid_rate','status_acc_strict','anomaly_balanced_acc','anomaly_recall_flagged','false_reassurance_rate',
        'unsupported_finding_rate_on_normals','defect_type_macro_f1','region_iou>=0.3_rate','latency_p50_s','latency_p95_s','peak_gpu_mem_gb']
rows = {}
for f in sorted(glob.glob(f'{DRIVE}/reports/vlm_eval/*_val_metrics.json')) + sorted(glob.glob('reports/vlm_eval/*_val_metrics.json')):
    m = json.load(open(f))
    if m.get('corrupt'): continue
    rows[os.path.basename(f).replace('_val_metrics.json', '')] = {k: m.get(k) for k in KEYS}   # local (newer) wins
df = pd.DataFrame(rows).T.sort_values('status_acc_strict', ascending=False)
gates = {'schema_valid_rate': ('>=', .995), 'anomaly_recall_flagged': ('>=', .85),
         'unsupported_finding_rate_on_normals': ('<=', .10), 'false_reassurance_rate': ('<=', .15)}
df['gates_pass'] = df.apply(lambda r: all((r[k] >= t) if op == '>=' else (r[k] <= t) for k, (op, t) in gates.items()), axis=1)
df.round(3)
"""),
        md("Zero-shot rows (no `lora` in the name) use a 10-per-cell sample, fine-tuned rows the full val split, so compare fine-tuned rows with each other. Latency is server-GPU latency; compare only rows measured on the same GPU type."),
        SYNC,
    ],
    "06_demo_inference_colab.ipynb": [
        md("# 06 · Try the fine-tuned SmolVLM-500M on a few images\n**Run all** (Runtime → Run all). About 3 minutes on a T4: no dataset download, the 8 demo images ship with the repo in `examples/` (val split: never trained on, never the locked test).\nGreen box = true defect region, red box = the model's predicted region. Section 3 lets you upload your own photo.\nNeeds the adapter trained in notebook 03 at `MyDrive/InspectEdge/checkpoints/<RUN>/final`."),
        code("""
import os
from google.colab import drive
drive.mount('/content/drive')
DRIVE, WORK = '/content/drive/MyDrive/InspectEdge', '/content/InspectEdge'
if os.path.isdir(f'{WORK}/.git'):
    !git -C {WORK} pull -q --ff-only
else:
    !git clone -q https://github.com/moghbaie/InspectEdge_Mobile_VLM.git {WORK}
%cd {WORK}
!pip -q install -r requirements-colab.txt && pip -q install -e . --no-deps
os.environ['INSPECTEDGE_ROOT'] = WORK
import sys; sys.path.insert(0, f'{WORK}/src')   # the editable install's .pth is only read at kernel start; in-kernel imports need this
import torch, transformers
assert transformers.__version__ == '4.57.1', 'pinned transformers not active: Runtime -> Restart session, then Run all again'
assert torch.cuda.is_available(), 'No GPU: Runtime -> Change runtime type -> T4 GPU'
"""),
        md("## 1. Load the model"),
        code("""
RUN = 'smolvlm500m_lora_r16_lr2e-4_s0'   # val winner (notebook 05, section 5); change to try another run
BASE = 'HuggingFaceTB/SmolVLM-500M-Instruct'

import io, json, time, matplotlib.pyplot as plt, matplotlib.patches as patches
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
from inspectedge.vlm.prompting import build_messages
from inspectedge.schemas.report import parse_model_output

ADAPTER, LONGEST_EDGE = f'{DRIVE}/checkpoints/{RUN}/final', 1024   # same image size as eval_vlm.py
assert os.path.isdir(ADAPTER), f'no adapter at {ADAPTER}: train it in notebook 03 first'
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
proc = AutoProcessor.from_pretrained(ADAPTER)
if 'longest_edge' in getattr(proc.image_processor, 'size', {}):
    proc.image_processor.size = {'longest_edge': LONGEST_EDGE}
model = AutoModelForImageTextToText.from_pretrained(BASE, dtype=dtype).to('cuda')
model = PeftModel.from_pretrained(model, ADAPTER).merge_and_unload().eval()   # merged: faster generation

def inspect(img, category):
    \"\"\"Image + category -> (ParseResult, raw text, seconds). Same prompt and greedy decoding as eval_vlm.py.\"\"\"
    prompt = proc.apply_chat_template(build_messages(category), add_generation_prompt=True)
    inputs = proc(text=[prompt], images=[[img]], return_tensors='pt').to(model.device)
    inputs['pixel_values'] = inputs['pixel_values'].to(dtype)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=320, do_sample=False)
    torch.cuda.synchronize()
    text = proc.batch_decode(out[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
    return parse_model_output(text), text, time.perf_counter() - t0

def draw(ax, boxes, size, color):
    W, H = size
    for b in boxes or []:
        if isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
            ax.add_patch(patches.Rectangle((b[0] * W, b[1] * H), (b[2] - b[0]) * W, (b[3] - b[1]) * H, fill=False, ec=color, lw=2))

def show(img, category, truth=None):
    \"\"\"Run the model and plot. truth = an examples.json entry (adds ground truth), or None for your own photo.\"\"\"
    pr, text, dt = inspect(img, category)
    rep = pr.parsed if pr.json_ok and isinstance(pr.parsed, dict) else {}
    finds = [f for f in rep['findings'] if isinstance(f, dict)] if isinstance(rep.get('findings'), list) else []
    status = rep.get('inspection_status', 'INVALID OUTPUT')
    fig, ax = plt.subplots(figsize=(5, 5)); ax.imshow(img); ax.axis('off')
    title, ok = f'{category} | model: {status}', None
    if truth:
        ok = (status == 'visible_anomaly') == (truth['label'] == 'anomaly') and pr.schema_ok
        title += f\"\\ntruth: {truth['label']} {truth['defect_types'] or ''} -> {'CORRECT' if ok else 'WRONG'}\"
        draw(ax, truth['boxes'], img.size, 'lime')
    draw(ax, [f.get('region_xyxy_normalized') for f in finds], img.size, 'red')
    ax.set_title(title, fontsize=10); plt.show()
    print(f'{dt:.1f}s | valid report: {pr.schema_ok}' + (f' | problems: {pr.errors[:2]}' if pr.errors else ''))
    for f in finds:
        print(f\"  - {f.get('type')}: {f.get('visible_evidence')} (review_required={f.get('review_required')})\")
    q = rep.get('image_quality') if isinstance(rep.get('image_quality'), dict) else {}
    if q: print('  image quality:', 'usable' if q.get('usable') else 'NOT usable', q.get('issues') or '')
    if not pr.json_ok: print('raw output:', text[:500])
    return ok
print('ready:', RUN)
"""),
        md("## 2. The 8 demo images (val split)"),
        code("""
EXAMPLES = json.load(open('examples/examples.json'))
results = [show(Image.open(f\"examples/{e['file']}\").convert('RGB'), e['category'], e) for e in EXAMPLES]
print(f'\\n{sum(results)}/{len(results)} correct on the demo images (full-val accuracy is in notebook 05, section 5)')
"""),
        md("Optional: the same defective PCB as a dark and a blurry photo. The model was trained with these corruptions and should report the quality problem rather than guess."),
        code("""
import numpy as np
from inspectedge.augmentation.corruptions import apply as corrupt
e = next(x for x in EXAMPLES if x['file'] == 'pcb1_anomaly.jpg')
arr = np.asarray(Image.open(f\"examples/{e['file']}\").convert('RGB'))
for fam in ['low_light', 'motion_blur']:
    print(f'===== {fam} =====')
    show(Image.fromarray(corrupt(arr, fam, 3, seed=0)[0]), e['category'], e)
"""),
        md("## 3. Your own photo\nSet `CATEGORY` to the closest of: candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum. The model only knows these 12 VisA objects photographed on a rig, so phone photos of other things are out of its training range."),
        code("""
from google.colab import files
CATEGORY = 'pcb1'
for name, data in files.upload().items():
    img = Image.open(io.BytesIO(data)).convert('RGB'); img.thumbnail((LONGEST_EDGE, LONGEST_EDGE))
    show(img, CATEGORY)
"""),
    ],
}


if __name__ == "__main__":
    import sys
    only = sys.argv[1:]   # e.g. `python scripts/make_notebooks.py 05` writes only notebooks whose name starts with 05
    (ROOT / "notebooks").mkdir(exist_ok=True)
    for name, cells in NB.items():
        if only and not name.startswith(tuple(only)):
            continue
        gpu = "A100" if name.startswith(A100_NBS) else "T4"
        (ROOT / "notebooks" / name).write_text(json.dumps(nb(cells, gpu), indent=1))
        print("wrote", name, len(cells), "cells")
