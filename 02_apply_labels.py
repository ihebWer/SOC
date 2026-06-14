#!/usr/bin/env python3
from pathlib import Path

import pandas as pd


EVENTS_FILE = Path("data/wazuh_native_events.parquet")
LABELS_FILE = Path("labels.csv")
OUTPUT_FILE = Path("data/wazuh_native_events_labeled.parquet")

CRITICAL_CLASSES = {
    "cracking",
    "webshell",
    "privilege_escalation",
    "reverse_shell",
}


def main():
    print("[load] events...")
    df = pd.read_parquet(EVENTS_FILE)

    print("[load] labels...")
    labels = pd.read_csv(LABELS_FILE)

    required = {"scenario", "attack", "start", "end"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans labels.csv : {missing}")

    labels["scenario"] = labels["scenario"].astype(str)
    labels["attack"] = labels["attack"].astype(str)
    labels["start"] = pd.to_numeric(labels["start"], errors="coerce").astype("int64")
    labels["end"] = pd.to_numeric(labels["end"], errors="coerce").astype("int64")

    df["time_label"] = "false_positive"

    print("[labeling] application des intervalles d'attaque...")

    total_marked = 0

    for scenario, g in labels.groupby("scenario"):
        mask_scenario = df["scenario"] == scenario

        if mask_scenario.sum() == 0:
            print(f"[warn] scenario absent dans events : {scenario}")
            continue

        for _, row in g.iterrows():
            attack = row["attack"]
            start = int(row["start"])
            end = int(row["end"])

            mask = (
                mask_scenario
                & (df["time"] >= start)
                & (df["time"] <= end)
            )

            n = int(mask.sum())

            if n > 0:
                df.loc[mask, "time_label"] = attack
                total_marked += n

            print(
                f"  {scenario:<18} {attack:<25} "
                f"{start} -> {end} : {n:,} alertes".replace(",", " ")
            )

    df["target"] = df["time_label"].isin(CRITICAL_CLASSES).astype("int8")

    print("\n[summary]")
    print("Events:", df.shape)
    print("Alertes labellisées attaque:", total_marked)

    print("\nDistribution time_label:")
    print(df["time_label"].value_counts())

    print("\nDistribution target:")
    print(df["target"].value_counts())

    print("\nDistribution par scénario:")
    print(pd.crosstab(df["scenario"], df["target"]))

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()