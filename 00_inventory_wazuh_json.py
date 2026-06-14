#!/usr/bin/env python3
# 00_inventory_wazuh_json.py
# But : comprendre EXACTEMENT ce que contiennent les fichiers *_wazuh.json
#       AVANT d'ecrire la moindre feature.
#
# Usage :
#   python3 00_inventory_wazuh_json.py            
#   python3 00_inventory_wazuh_json.py 
#
# Produit :
#   - un rapport lisible dans le terminal
#   - wazuh_samples.txt  : 2 alertes completes par scenario (a me copier-coller)

import json, sys, glob, os, time
from collections import Counter

LABEL_HINTS = ("label", "attack", "ground", "truth", "malicious",
               "classif", "is_attack", "tag")  # on cherche la verite terrain

def dig(obj, *path, default=None):
    """Acces defensif a un champ imbrique : dig(r,'rule','groups')."""
    cur = obj
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def detect_format(path):
    """Renvoie 'jsonl' ou 'array' en lisant le 1er caractere non vide."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        while True:
            ch = f.read(1)
            if ch == "":
                return "jsonl"          # fichier vide, peu importe
            if not ch.isspace():
                return "array" if ch == "[" else "jsonl"

def iter_records(path):
    """Genere les alertes une par une, quel que soit le format."""
    fmt = detect_format(path)
    if fmt == "array":
        # tableau JSON : on charge en bloc (rare pour du Wazuh)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            try:
                data = json.load(f)
                for rec in data:
                    yield rec
                return
            except Exception:
                pass  # on retombe sur le mode ligne ci-dessous
    # mode JSONL : une alerte par ligne
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line or line in ("[", "]"):
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue  # ligne illisible : on la compte ailleurs

def is_suricata(r):
    dname = (dig(r, "decoder", "name") or "").lower()
    if "suricata" in dname:
        return True
    if dig(r, "data", "alert") is not None:
        return True
    if dig(r, "data", "event_type") == "alert":
        return True
    return False

def get_srcip(r):
    return dig(r, "data", "srcip") or dig(r, "data", "src_ip")

def get_dstip(r):
    return dig(r, "data", "dstip") or dig(r, "data", "dest_ip") or dig(r, "data", "dstip")

def analyze_file(path, samples_fh):
    t0 = time.time()
    size_mb = os.path.getsize(path) / 1e6
    name = os.path.basename(path)
    print(f"\n{'='*70}\n{name}   ({size_mb:.1f} Mo)\n{'='*70}", flush=True)

    n = 0
    rid       = Counter()
    rdesc     = Counter()
    rgroups   = Counter()
    rdecoder  = Counter()
    rlevel    = Counter()
    topkeys   = Counter()       # cles de haut niveau (le schema)
    datakeys  = Counter()       # cles sous data.*
    n_mitre = n_suri = n_src = n_dst = 0
    label_keys = {}             # cle suspecte -> exemple de valeur
    samples = []

    for r in iter_records(path):
        if not isinstance(r, dict):
            continue
        n += 1

        for k in r.keys():
            topkeys[k] += 1
        for k in (dig(r, "data") or {}):
            datakeys[k] += 1

        if dig(r, "rule", "id") is not None:
            rid[str(dig(r, "rule", "id"))] += 1
        if dig(r, "rule", "description"):
            rdesc[dig(r, "rule", "description")] += 1
        for g in (dig(r, "rule", "groups") or []):
            rgroups[g] += 1
        if dig(r, "decoder", "name"):
            rdecoder[dig(r, "decoder", "name")] += 1
        if dig(r, "rule", "level") is not None:
            rlevel[str(dig(r, "rule", "level"))] += 1

        mitre = dig(r, "rule", "mitre")
        if isinstance(mitre, dict) and any(mitre.values()):
            n_mitre += 1
        if is_suricata(r):
            n_suri += 1
        if get_srcip(r):
            n_src += 1
        if get_dstip(r):
            n_dst += 1

        # chasse au label : on inspecte les noms de cles (haut niveau + data)
        for k in list(r.keys()) + ["data." + dk for dk in (dig(r, "data") or {})]:
            low = k.lower()
            if any(h in low for h in LABEL_HINTS) and k not in label_keys:
                if k.startswith("data."):
                    label_keys[k] = dig(r, "data", k[5:])
                else:
                    label_keys[k] = r.get(k)

        if len(samples) < 2:
            samples.append(r)

        if n % 200000 == 0:
            print(f"  ... {n} alertes lues", flush=True)

    # ----- rapport -----
    print(f"Nombre d'alertes      : {n}")
    print(f"Temps de lecture      : {time.time()-t0:.1f} s")
    print(f"\nCles de haut niveau   : {sorted(topkeys)}")
    print(f"Cles sous data.*      : {sorted(datakeys)[:40]}")

    def show(title, counter, k=10):
        print(f"\n{title} (top {k}) :")
        for val, c in counter.most_common(k):
            print(f"   {c:>8}  {val}")

    show("rule.id", rid)
    show("rule.description", rdesc)
    show("rule.groups", rgroups)
    show("decoder.name", rdecoder)
    print(f"\nrule.level (distribution) :")
    for lvl, c in sorted(rlevel.items(), key=lambda x: int(x[0]) if x[0].isdigit() else -1):
        print(f"   niveau {lvl:>3} : {c}")

    pct = lambda x: f"{100*x/n:.1f}%" if n else "0%"
    print(f"\nPresence MITRE        : {n_mitre} ({pct(n_mitre)})")
    print(f"Presence Suricata     : {n_suri} ({pct(n_suri)})")
    print(f"Presence src_ip       : {n_src} ({pct(n_src)})")
    print(f"Presence dst_ip       : {n_dst} ({pct(n_dst)})")

    print(f"\n>>> CHAMP LABEL / VERITE TERRAIN :")
    if label_keys:
        for k, v in label_keys.items():
            print(f"   TROUVE  '{k}'  exemple = {v!r}")
    else:
        print("   AUCUN champ label detecte dans le JSON.")
        print("   => les labels devront etre recuperes par jointure (AIT-LDS / .txt).")

    # echantillons complets pour analyse de structure
    samples_fh.write(f"\n\n{'#'*70}\n# {name}\n{'#'*70}\n")
    for i, s in enumerate(samples, 1):
        samples_fh.write(f"\n--- alerte exemple {i} ---\n")
        samples_fh.write(json.dumps(s, indent=2, ensure_ascii=False))
        samples_fh.write("\n")

    return {"scenario": name, "n": n, "mitre": n_mitre, "suricata": n_suri,
            "src": n_src, "dst": n_dst, "has_label": bool(label_keys)}

def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(glob.glob(os.path.join(folder, "*_wazuh.json")))
    if not files:
        print(f"Aucun fichier *_wazuh.json dans {folder!r}")
        sys.exit(1)

    print(f"{len(files)} fichiers Wazuh trouves.")
    results = []
    with open("wazuh_samples.txt", "w", encoding="utf-8") as sfh:
        for path in files:
            results.append(analyze_file(path, sfh))

    print(f"\n\n{'='*70}\nSYNTHESE GLOBALE\n{'='*70}")
    total = sum(r["n"] for r in results)
    print(f"Total alertes (8 scenarios) : {total}")
    print(f"{'scenario':<26}{'alertes':>10}{'MITRE':>8}{'Suri':>8}{'label':>8}")
    for r in results:
        print(f"{r['scenario']:<26}{r['n']:>10}{r['mitre']:>8}{r['suricata']:>8}"
              f"{('oui' if r['has_label'] else 'non'):>8}")
    print(f"\nEchantillons complets ecrits dans : wazuh_samples.txt")

if __name__ == "__main__":
    main()