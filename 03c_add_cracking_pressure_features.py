#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

INPUT = Path("data/wazuh_native_features_v2.parquet")
OUTPUT = Path("data/wazuh_native_features_v3.parquet")

EPS = 1e-6

print("=" * 80)
print("ADD CRACKING PRESSURE FEATURES — WAZUH NATIVE V3")
print("=" * 80)

df = pd.read_parquet(INPUT)
print("[load]", df.shape)

new = pd.DataFrame(index=df.index)

# 1. Volume long terme vs court terme
new["alert_long_pressure_1h_vs_5m"] = (
    df["alerts_total_3600s"] / (df["alerts_total_300s"] + 1)
).astype(np.float32)

new["alert_long_pressure_1h_vs_1m"] = (
    df["alerts_total_3600s"] / (df["alerts_total_60s"] + 1)
).astype(np.float32)

new["alert_mid_pressure_15m_vs_1m"] = (
    df["alerts_total_900s"] / (df["alerts_total_60s"] + 1)
).astype(np.float32)

# 2. Suricata long-term dominance
if "suricata_signature_count_3600s" in df.columns:
    new["suricata_long_pressure"] = (
        df["suricata_signature_count_3600s"] / (df["suricata_signature_count_300s"] + 1)
    ).astype(np.float32)

    new["suricata_volume_ratio_1h"] = (
        df["suricata_signature_count_3600s"] / (df["alerts_total_3600s"] + 1)
    ).astype(np.float32)

# 3. MITRE lateral movement density
if "mitre_lateral_count_3600s" in df.columns:
    new["mitre_lateral_density_1h"] = (
        df["mitre_lateral_count_3600s"] / (df["alerts_total_3600s"] + 1)
    ).astype(np.float32)

    new["mitre_lateral_pressure_1h_vs_15m"] = (
        df["mitre_lateral_count_3600s"] / (df["mitre_lateral_count_900s"] + 1)
    ).astype(np.float32)

# 4. Auth vs IDS balance
new["auth_ids_balance_1h"] = (
    df["auth_ratio_3600s"] / (df["ids_ratio_3600s"] + EPS)
).astype(np.float32)

new["auth_ids_balance_5m"] = (
    df["auth_ratio_300s"] / (df["ids_ratio_300s"] + EPS)
).astype(np.float32)

# 5. Diversity pressure
new["rule_diversity_growth_1h_vs_5m"] = (
    df["unique_rule_ids_3600s"] / (df["unique_rule_ids_300s"] + 1)
).astype(np.float32)

new["srcip_focus_change_1h_vs_5m"] = (
    df["srcip_alert_ratio_300s"] / (df["srcip_alert_ratio_3600s"] + EPS)
).astype(np.float32)

# 6. Composite cracking pressure score
parts = []

for col in [
    "alert_long_pressure_1h_vs_5m",
    "suricata_long_pressure",
    "mitre_lateral_pressure_1h_vs_15m",
    "auth_ids_balance_1h",
    "rule_diversity_growth_1h_vs_5m",
]:
    if col in new.columns:
        x = new[col].replace([np.inf, -np.inf], 0).fillna(0)
        x = np.log1p(x.clip(lower=0))
        parts.append(x)

if parts:
    new["cracking_pressure_score"] = (
        pd.concat(parts, axis=1).mean(axis=1)
    ).astype(np.float32)

new = new.replace([np.inf, -np.inf], 0).fillna(0)

out = pd.concat([df.reset_index(drop=True), new.reset_index(drop=True)], axis=1)

print("[new features]", new.shape)
for c in new.columns:
    print(" -", c)

print("[final]", out.shape)

out.to_parquet(OUTPUT, index=False)
print("Saved:", OUTPUT)