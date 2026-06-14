#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

EVENTS_FILE = Path("data/wazuh_native_events_labeled.parquet")
FEATURES_FILE = Path("data/wazuh_native_features.parquet")
OUTPUT_FILE = Path("data/wazuh_native_features_v2.parquet")

WINDOWS = [60, 300, 900, 3600]


def clean_str(s):
    return s.fillna("").astype(str).replace({"nan": "", "None": "", "NaN": ""})


def main():
    print("=" * 80)
    print("ADD RULE CONTEXT FEATURES — WAZUH NATIVE V2 CLEAN")
    print("=" * 80)

    events = pd.read_parquet(EVENTS_FILE)
    features = pd.read_parquet(FEATURES_FILE)

    events = events.sort_values(["scenario", "time"]).reset_index(drop=True)
    features = features.sort_values(["scenario", "time"]).reset_index(drop=True)

    assert len(events) == len(features)
    assert (events["scenario"].values == features["scenario"].values).all()
    assert (events["time"].values == features["time"].values).all()

    events["rule_id"] = clean_str(events["rule_id"])
    events["src_ip"] = clean_str(events["src_ip"])
    events["agent_name"] = clean_str(events["agent_name"])
    events["suricata_signature_id"] = clean_str(events["suricata_signature_id"])
    events["mitre_tactics"] = clean_str(events["mitre_tactics"])
    events["mitre_techniques"] = clean_str(events["mitre_techniques"])

    new_parts = []

    has_mitre = (
        (events["mitre_tactics"].str.len() > 0) |
        (events["mitre_techniques"].str.len() > 0)
    ).astype("int8")

    mitre_tactic_count = events["mitre_tactics"].apply(
        lambda x: len([p for p in str(x).split("|") if p])
    ).astype("int16")

    mitre_technique_count = events["mitre_techniques"].apply(
        lambda x: len([p for p in str(x).split("|") if p])
    ).astype("int16")

    new_parts.append(pd.DataFrame({
        "has_mitre": has_mitre,
        "mitre_tactic_count": mitre_tactic_count,
        "mitre_technique_count": mitre_technique_count,
    }))

    for w in WINDOWS:
        print(f"[window {w}s]")
        bucket = f"bucket_{w}s"
        events[bucket] = (events["time"] // w) * w

        base = events[[
            "scenario", bucket, "rule_id", "src_ip",
            "agent_name", "suricata_signature_id"
        ]].copy()

        grule = events.groupby(
            ["scenario", bucket, "rule_id"], observed=True
        ).size().reset_index(name=f"rule_count_{w}s")

        grule_ip = events.groupby(
            ["scenario", bucket, "src_ip", "rule_id"], observed=True
        ).size().reset_index(name=f"rule_srcip_count_{w}s")

        grule_agent = events.groupby(
            ["scenario", bucket, "agent_name", "rule_id"], observed=True
        ).size().reset_index(name=f"rule_agent_count_{w}s")

        gsig = events.groupby(
            ["scenario", bucket, "suricata_signature_id"], observed=True
        ).size().reset_index(name=f"suricata_signature_count_{w}s")

        tmp = base.merge(grule, on=["scenario", bucket, "rule_id"], how="left")
        tmp = tmp.merge(grule_ip, on=["scenario", bucket, "src_ip", "rule_id"], how="left")
        tmp = tmp.merge(grule_agent, on=["scenario", bucket, "agent_name", "rule_id"], how="left")
        tmp = tmp.merge(gsig, on=["scenario", bucket, "suricata_signature_id"], how="left")

        tmp = tmp[[
            f"rule_count_{w}s",
            f"rule_srcip_count_{w}s",
            f"rule_agent_count_{w}s",
            f"suricata_signature_count_{w}s",
        ]].fillna(0).astype(np.float32)

        new_parts.append(tmp)

    new_df = pd.concat(new_parts, axis=1)

    for prefix in [
        "rule_count",
        "rule_srcip_count",
        "rule_agent_count",
        "suricata_signature_count",
    ]:
        new_df[f"{prefix}_burst_60_vs_3600"] = (
            new_df[f"{prefix}_60s"] /
            (new_df[f"{prefix}_3600s"] / 60.0 + 1.0)
        ).astype(np.float32)

        new_df[f"{prefix}_burst_300_vs_3600"] = (
            new_df[f"{prefix}_300s"] /
            (new_df[f"{prefix}_3600s"] / 12.0 + 1.0)
        ).astype(np.float32)

    out = pd.concat(
        [features.reset_index(drop=True), new_df.reset_index(drop=True)],
        axis=1
    ).fillna(0)

    print("Base :", features.shape)
    print("New  :", new_df.shape)
    print("Final:", out.shape)

    out.to_parquet(OUTPUT_FILE, index=False)
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()