#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import datetime

import pandas as pd


INPUT_DIR = Path(".")
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "wazuh_native_events.parquet"


def parse_time(value):
    if not value:
        return None

    value = str(value).replace("Z", "+00:00")

    try:
        return int(datetime.fromisoformat(value).timestamp())
    except Exception:
        return None


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def get_src_ip(alert, data, agent):
    return (
        data.get("src_ip")
        or data.get("srcip")
        or data.get("src_ip_address")
        or agent.get("ip")
        or ""
    )


def get_dst_ip(data):
    return (
        data.get("dest_ip")
        or data.get("dstip")
        or data.get("dst_ip")
        or ""
    )


def normalize_file(path):
    scenario = path.stem.replace("_wazuh", "")
    rows = []

    print(f"[load] {path.name}")

    with open(path, "r", errors="ignore") as f:
        for i, line in enumerate(f, 1):
            try:
                a = json.loads(line)
            except Exception:
                continue

            rule = a.get("rule", {}) or {}
            data = a.get("data", {}) or {}
            decoder = a.get("decoder", {}) or {}
            agent = a.get("agent", {}) or {}
            manager = a.get("manager", {}) or {}

            mitre = rule.get("mitre", {}) or {}
            suri_alert = data.get("alert", {}) if isinstance(data.get("alert"), dict) else {}

            ts = (
                a.get("@timestamp")
                or a.get("timestamp")
                or data.get("timestamp")
            )

            t = parse_time(ts)
            if t is None:
                continue

            groups = rule.get("groups", [])
            if not isinstance(groups, list):
                groups = []

            tactics = mitre.get("tactic", [])
            techniques = mitre.get("technique", [])

            if not isinstance(tactics, list):
                tactics = [str(tactics)]
            if not isinstance(techniques, list):
                techniques = [str(techniques)]

            src_ip = get_src_ip(a, data, agent)
            dst_ip = get_dst_ip(data)

            rows.append({
                "scenario": scenario,
                "time": t,

                "agent_id": agent.get("id", ""),
                "agent_name": agent.get("name", ""),
                "agent_ip": agent.get("ip", ""),
                "manager_name": manager.get("name", ""),

                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_port": str(data.get("src_port") or data.get("srcport") or ""),
                "dst_port": str(data.get("dest_port") or data.get("dstport") or ""),

                "proto": str(data.get("proto") or data.get("protocol") or ""),
                "app_proto": str(data.get("app_proto") or data.get("app_proto_tc") or ""),
                "event_type": str(data.get("event_type") or ""),

                "decoder_name": decoder.get("name", ""),
                "decoder_parent": decoder.get("parent", ""),

                "rule_id": str(rule.get("id", "")),
                "rule_level": safe_int(rule.get("level", 0)),
                "rule_description": rule.get("description", ""),
                "rule_groups": "|".join(groups),

                "mitre_tactics": "|".join(tactics),
                "mitre_techniques": "|".join(techniques),

                "suricata_signature_id": str(suri_alert.get("signature_id", "")),
                "suricata_signature": suri_alert.get("signature", ""),
                "suricata_category": suri_alert.get("category", ""),
                "suricata_severity": safe_int(suri_alert.get("severity", 0)),
                "suricata_action": suri_alert.get("action", ""),

                "location": a.get("location", ""),
                "full_log": a.get("full_log", ""),
            })

            if i % 200000 == 0:
                print(f"  ... {i} lignes lues")

    return rows


def main():
    files = sorted(INPUT_DIR.glob("*_wazuh.json"))

    if not files:
        raise FileNotFoundError("Aucun fichier *_wazuh.json trouvé dans le dossier courant.")

    all_rows = []

    for path in files:
        rows = normalize_file(path)
        print(f"  -> {len(rows):,} alertes normalisées".replace(",", " "))
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    df = df.sort_values(["scenario", "time"]).reset_index(drop=True)

    print("\nDataset normalisé :", df.shape)
    print("\nScénarios :")
    print(df["scenario"].value_counts())

    print("\nTop rule_id :")
    print(df["rule_id"].value_counts().head(20))

    print("\nTop rule_groups :")
    print(df["rule_groups"].value_counts().head(20))

    print("\nRule level :")
    print(df["rule_level"].value_counts().sort_index())

    df.to_parquet(OUTPUT_FILE, index=False)

    print("\nSauvegardé :", OUTPUT_FILE)


if __name__ == "__main__":
    main()