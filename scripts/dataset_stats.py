"""Data sufficiency report: counts per split/category/defect type, defect size, image stats."""
import json
from collections import Counter

import numpy as np
import pandas as pd

from inspectedge.paths import ROOT

m = pd.read_json(ROOT / "data/manifests/visa_manifest.jsonl", lines=True)
a = m[m.label == "anomaly"]
out = {}
out["by_split_label"] = m.groupby(["vlm_split", "label"]).size().unstack().to_dict()
out["anomalies_by_category_split"] = a.groupby(["category", "vlm_split"]).size().unstack().fillna(0).astype(int).to_dict()
prim = a.assign(primary=a.defect_types.str[0])
out["primary_type_by_split"] = prim.groupby(["primary", "vlm_split"]).size().unstack().fillna(0).astype(int).to_dict()
allt = Counter(t for ts in a.defect_types for t in ts)
out["all_type_mentions"] = dict(allt)
out["mask_area_frac_pct"] = dict(zip(["p5", "p25", "p50", "p75", "p95"], np.percentile(a.mask_area_frac, [5, 25, 50, 75, 95]).round(5).tolist()))
out["boxes_per_anomaly"] = a.boxes.apply(len).describe().round(2).to_dict()
out["image_size_px"] = m.groupby("category")[["width", "height"]].median().astype(int).to_dict()
out["sharpness_lap_var_pct"] = np.percentile(m.lap_var, [5, 50, 95]).round(1).tolist()
out["mean_luma_pct"] = np.percentile(m.mean_luma, [5, 50, 95]).round(1).tolist()
out["pcb_subset"] = m[m.asset_family == "electronics_pcb"].groupby(["vlm_split", "label"]).size().unstack().to_dict()
(ROOT / "reports/dataset_stats.json").write_text(json.dumps(out, indent=2))
print(pd.DataFrame(out["by_split_label"]).to_string())
print(pd.DataFrame(out["anomalies_by_category_split"]).to_string())
print(pd.DataFrame(out["primary_type_by_split"]).to_string())
for k in ["all_type_mentions", "mask_area_frac_pct", "boxes_per_anomaly", "sharpness_lap_var_pct", "mean_luma_pct", "pcb_subset"]:
    print(k, out[k])
