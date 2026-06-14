#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

INPUT = Path("data/wazuh_native_events_labeled.parquet")
OUTPUT = Path("data/wazuh_native_features.parquet")

WINDOWS = [60, 300, 900, 3600]

def contains(s, word):
    return s.fillna("").str.lower().str.contains(word, regex=False)

def main():
    print("[load]")
    df = pd.read_parquet(INPUT)
    df = df.sort_values(["scenario", "time"]).reset_index(drop=True)

    print("[categorize]")
    groups = df["rule_groups"].fillna("").str.lower()
    desc = df["rule_description"].fillna("").str.lower()
    decoder = df["decoder_name"].fillna("").str.lower()
    app = df["app_proto"].fillna("").str.lower()
    sig = df["suricata_signature"].fillna("").str.lower()
    cat = df["suricata_category"].fillna("").str.lower()
    mitre_t = df["mitre_tactics"].fillna("").str.lower()

    df["is_ids"] = groups.str.contains("ids").astype("int8")
    df["is_suricata"] = groups.str.contains("suricata").astype("int8")
    df["is_web"] = groups.str.contains("web").astype("int8")
    df["is_auth"] = (
        groups.str.contains("authentication") |
        groups.str.contains("pam") |
        groups.str.contains("sshd") |
        groups.str.contains("dovecot")
    ).astype("int8")
    df["is_attack_group"] = groups.str.contains("attack").astype("int8")
    df["is_recon_group"] = groups.str.contains("recon").astype("int8")
    df["is_dns"] = ((app == "dns") | desc.str.contains("dns") | sig.str.contains("dns")).astype("int8")
    df["is_tls"] = ((app == "tls") | desc.str.contains("tls") | sig.str.contains("tls")).astype("int8")
    df["is_http"] = ((app == "http") | desc.str.contains("http") | decoder.str.contains("web")).astype("int8")
    df["is_policy"] = (sig.str.contains("policy") | cat.str.contains("policy")).astype("int8")
    df["is_malware"] = (
        groups.str.contains("virus") |
        desc.str.contains("clamav") |
        sig.str.contains("malware")
    ).astype("int8")

    df["mitre_initial_access"] = mitre_t.str.contains("initial access").astype("int8")
    df["mitre_execution"] = mitre_t.str.contains("execution").astype("int8")
    df["mitre_privilege"] = mitre_t.str.contains("privilege escalation").astype("int8")
    df["mitre_lateral"] = mitre_t.str.contains("lateral movement").astype("int8")
    df["mitre_discovery"] = mitre_t.str.contains("discovery").astype("int8")

    df["high_level"] = (df["rule_level"] >= 10).astype("int8")
    df["medium_level"] = ((df["rule_level"] >= 6) & (df["rule_level"] < 10)).astype("int8")

    cat_cols = [
        "is_ids", "is_suricata", "is_web", "is_auth", "is_attack_group",
        "is_recon_group", "is_dns", "is_tls", "is_http", "is_policy",
        "is_malware", "mitre_initial_access", "mitre_execution",
        "mitre_privilege", "mitre_lateral", "mitre_discovery",
        "high_level", "medium_level"
    ]

    feature_parts = []
    base_cols = ["scenario", "time", "time_label", "target", "src_ip", "agent_name", "rule_id", "rule_level"]
    out = df[base_cols].copy()

    for w in WINDOWS:
        print(f"[window {w}s]")
        bucket = f"bucket_{w}s"
        df[bucket] = (df["time"] // w) * w

        # Global scenario/bucket
        g = df.groupby(["scenario", bucket], observed=True)
        global_agg = g.agg(
            alerts_total=("rule_id", "size"),
            unique_rule_ids=("rule_id", "nunique"),
            unique_src_ips=("src_ip", "nunique"),
            unique_agents=("agent_name", "nunique"),
            mean_rule_level=("rule_level", "mean"),
            max_rule_level=("rule_level", "max"),
        ).reset_index()

        for c in cat_cols:
            tmp = g[c].sum().reset_index(name=f"{c}_count")
            global_agg = global_agg.merge(tmp, on=["scenario", bucket], how="left")

        global_agg = global_agg.rename(columns={
            "alerts_total": f"alerts_total_{w}s",
            "unique_rule_ids": f"unique_rule_ids_{w}s",
            "unique_src_ips": f"unique_src_ips_{w}s",
            "unique_agents": f"unique_agents_{w}s",
            "mean_rule_level": f"mean_rule_level_{w}s",
            "max_rule_level": f"max_rule_level_{w}s",
        })

        for c in cat_cols:
            global_agg = global_agg.rename(columns={f"{c}_count": f"{c}_count_{w}s"})

        # src_ip/bucket
        gip = df.groupby(["scenario", "src_ip", bucket], observed=True)
        ip_agg = gip.agg(
            srcip_alerts=("rule_id", "size"),
            srcip_unique_rules=("rule_id", "nunique"),
            srcip_mean_rule_level=("rule_level", "mean"),
            srcip_max_rule_level=("rule_level", "max"),
        ).reset_index()

        for c in ["is_ids", "is_suricata", "is_web", "is_auth", "is_dns", "is_tls", "high_level"]:
            tmp = gip[c].sum().reset_index(name=f"srcip_{c}_count")
            ip_agg = ip_agg.merge(tmp, on=["scenario", "src_ip", bucket], how="left")

        ip_agg = ip_agg.rename(columns={
            "srcip_alerts": f"srcip_alerts_{w}s",
            "srcip_unique_rules": f"srcip_unique_rules_{w}s",
            "srcip_mean_rule_level": f"srcip_mean_rule_level_{w}s",
            "srcip_max_rule_level": f"srcip_max_rule_level_{w}s",
        })

        for c in ["is_ids", "is_suricata", "is_web", "is_auth", "is_dns", "is_tls", "high_level"]:
            ip_agg = ip_agg.rename(columns={f"srcip_{c}_count": f"srcip_{c}_count_{w}s"})

        # agent/bucket
        ga = df.groupby(["scenario", "agent_name", bucket], observed=True)
        agent_agg = ga.agg(
            agent_alerts=("rule_id", "size"),
            agent_unique_rules=("rule_id", "nunique"),
            agent_mean_rule_level=("rule_level", "mean"),
            agent_max_rule_level=("rule_level", "max"),
        ).reset_index()

        for c in ["is_ids", "is_suricata", "is_web", "is_auth", "is_dns", "is_tls", "high_level"]:
            tmp = ga[c].sum().reset_index(name=f"agent_{c}_count")
            agent_agg = agent_agg.merge(tmp, on=["scenario", "agent_name", bucket], how="left")

        agent_agg = agent_agg.rename(columns={
            "agent_alerts": f"agent_alerts_{w}s",
            "agent_unique_rules": f"agent_unique_rules_{w}s",
            "agent_mean_rule_level": f"agent_mean_rule_level_{w}s",
            "agent_max_rule_level": f"agent_max_rule_level_{w}s",
        })

        for c in ["is_ids", "is_suricata", "is_web", "is_auth", "is_dns", "is_tls", "high_level"]:
            agent_agg = agent_agg.rename(columns={f"agent_{c}_count": f"agent_{c}_count_{w}s"})

        tmp = df[["scenario", "time", "src_ip", "agent_name", bucket]].copy()
        tmp = tmp.merge(global_agg, on=["scenario", bucket], how="left")
        tmp = tmp.merge(ip_agg, on=["scenario", "src_ip", bucket], how="left")
        tmp = tmp.merge(agent_agg, on=["scenario", "agent_name", bucket], how="left")

        tmp = tmp.drop(columns=[bucket, "scenario", "time", "src_ip", "agent_name"])
        feature_parts.append(tmp)

    features = pd.concat([out] + feature_parts, axis=1)
    features = features.fillna(0)

    for w in WINDOWS:
        features[f"srcip_alert_ratio_{w}s"] = (
            features[f"srcip_alerts_{w}s"] / features[f"alerts_total_{w}s"].clip(lower=1)
        )
        features[f"agent_alert_ratio_{w}s"] = (
            features[f"agent_alerts_{w}s"] / features[f"alerts_total_{w}s"].clip(lower=1)
        )
        features[f"high_level_ratio_{w}s"] = (
            features[f"high_level_count_{w}s"] / features[f"alerts_total_{w}s"].clip(lower=1)
        )
        features[f"ids_ratio_{w}s"] = (
            features[f"is_ids_count_{w}s"] / features[f"alerts_total_{w}s"].clip(lower=1)
        )
        features[f"auth_ratio_{w}s"] = (
            features[f"is_auth_count_{w}s"] / features[f"alerts_total_{w}s"].clip(lower=1)
        )
        features[f"dns_ratio_{w}s"] = (
            features[f"is_dns_count_{w}s"] / features[f"alerts_total_{w}s"].clip(lower=1)
        )

    print("[save]")
    print("Features shape:", features.shape)
    print("Target distribution:")
    print(features["target"].value_counts())
    print("Time label distribution:")
    print(features["time_label"].value_counts())

    features.to_parquet(OUTPUT, index=False)
    print("Saved:", OUTPUT)

if __name__ == "__main__":
    main()