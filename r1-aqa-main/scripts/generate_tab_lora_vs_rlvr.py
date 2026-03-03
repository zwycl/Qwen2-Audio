#!/usr/bin/env python3
"""
Generate data for Table: Per-sample comparison of 20% RLVR vs 100% LoRA.
Finds compelling examples where each model wins, plus aggregate win counts.
Uses models from Table 1: 20% -CER+refine+rew (RLVR) and 100% LoRA SFT.
"""

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

EVAL_DIR = Path("/home/ubuntu/Qwen2-Audio/r1-aqa-main/outputs/eval_results")

# ── Per-sample eval files (seed 42) ─────────────────────────────────────────
FILES = {
    # SwitchLingua
    "20% RLVR (SL)": EVAL_DIR / "eval_switchlingua_csfleurs_xtts_train_cgpr_plus_n4625_e4_twostep_novad_s42_n999999_20260223_231912.json",
    "100% LoRA (SL)": EVAL_DIR / "eval_switchlingua_lora_csfleurs_xtts_train_n999999_e8_s42_n999999_20260224_070711.json",
    # CS-FLEURS read_test
    "20% RLVR (CF)": EVAL_DIR / "eval_csfleurs_csfleurs_xtts_train_cgpr_plus_n4625_e4_twostep_novad_s42_read_test_n999999_20260223_224314.json",
    "100% LoRA (CF)": EVAL_DIR / "eval_csfleurs_lora_csfleurs_xtts_train_n999999_e8_s42_read_test_n999999_20260224_061104.json",
}

CER_GAP = 0.2       # minimum CER difference to count as a "win"
MAX_REF_LEN = 150    # max reference length for example selection
TOP_EXAMPLES = 6     # examples to show per category


def get_uid(r: dict) -> str:
    return r.get("uniq_id", f"{r.get('language', r.get('language_pair',''))}_{r.get('index','')}")


def get_lang(r: dict) -> str:
    return r.get("language", r.get("language_pair", ""))


def has_repetition(text: str) -> bool:
    return bool(re.search(r"(.{5,})\1{2,}", text))


def compare(rlvr_data, lora_data, dataset_label):
    rlvr_by_id = {get_uid(r): r for r in rlvr_data["results"]}
    lora_by_id = {get_uid(r): r for r in lora_data["results"]}
    common = set(rlvr_by_id) & set(lora_by_id)

    rlvr_wins = []
    lora_wins = []

    for uid in common:
        rr = rlvr_by_id[uid]
        lr = lora_by_id[uid]
        diff = lr["cer"] - rr["cer"]  # positive → RLVR better
        lang = get_lang(rr)
        if diff > CER_GAP:
            rlvr_wins.append((uid, rr, lr, diff, lang))
        elif diff < -CER_GAP:
            lora_wins.append((uid, rr, lr, diff, lang))

    # ── Aggregate stats ──────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"{dataset_label}: 20% RLVR vs 100% LoRA")
    print(f"{'='*72}")
    print(f"Common samples: {len(common)}")
    print(f"RLVR wins (>{CER_GAP} CER gap): {len(rlvr_wins)}")
    print(f"LoRA wins (>{CER_GAP} CER gap): {len(lora_wins)}")

    # Per-language
    rlvr_lang = defaultdict(int)
    lora_lang = defaultdict(int)
    for *_, lang in rlvr_wins:
        rlvr_lang[lang] += 1
    for *_, lang in lora_wins:
        lora_lang[lang] += 1

    all_langs = sorted(set(rlvr_lang) | set(lora_lang))
    print(f"\n{'Language':12s} {'RLVR wins':>10s} {'LoRA wins':>10s}")
    print("-" * 35)
    for lang in all_langs:
        print(f"{lang:12s} {rlvr_lang[lang]:10d} {lora_lang[lang]:10d}")

    # ── CER distribution ─────────────────────────────────────────────────
    rlvr_cers = [r["cer"] for r in rlvr_data["results"]]
    lora_cers = [r["cer"] for r in lora_data["results"]]
    n = len(rlvr_cers)

    rlvr_perfect = sum(1 for c in rlvr_cers if c < 0.05)
    lora_perfect = sum(1 for c in lora_cers if c < 0.05)
    rlvr_bad = sum(1 for c in rlvr_cers if c > 0.5)
    lora_bad = sum(1 for c in lora_cers if c > 0.5)

    print(f"\nCER distribution:")
    print(f"  RLVR: <0.05 = {rlvr_perfect}/{n} ({100*rlvr_perfect/n:.1f}%), "
          f">0.5 = {rlvr_bad}/{n} ({100*rlvr_bad/n:.1f}%)")
    print(f"  LoRA: <0.05 = {lora_perfect}/{n} ({100*lora_perfect/n:.1f}%), "
          f">0.5 = {lora_bad}/{n} ({100*lora_bad/n:.1f}%)")

    # ── Repetition detection ─────────────────────────────────────────────
    rlvr_rep = sum(1 for r in rlvr_data["results"] if has_repetition(r["prediction"]))
    lora_rep = sum(1 for r in lora_data["results"] if has_repetition(r["prediction"]))
    print(f"\nRepetition loops: RLVR={rlvr_rep}, LoRA={lora_rep}")

    # ── Example selection ────────────────────────────────────────────────
    # RLVR wins: prefer short references, typologically distant pairs
    priority_langs = {"Arabic", "Japanese", "Korean", "Russian", "Mandarin",
                      "ara-eng", "jpn-eng", "kor-eng", "rus-eng", "cmn-eng"}
    rlvr_wins.sort(key=lambda x: -x[3])
    lora_wins.sort(key=lambda x: x[3])

    print(f"\n── RLVR wins (LoRA fails) ──")
    shown = 0
    for uid, rr, lr, diff, lang in rlvr_wins:
        ref = rr["reference"]
        if len(ref) <= MAX_REF_LEN and lang in priority_langs:
            print(f"\n  [{lang}] {uid}  (CER gap = {diff:.2f})")
            print(f"  REF:  {ref}")
            print(f"  RLVR: {rr['prediction'][:200]}  (CER={rr['cer']:.3f})")
            print(f"  LoRA: {lr['prediction'][:200]}  (CER={lr['cer']:.3f})")
            shown += 1
            if shown >= TOP_EXAMPLES:
                break

    print(f"\n── LoRA wins (RLVR fails) ──")
    shown = 0
    for uid, rr, lr, diff, lang in lora_wins:
        ref = rr["reference"]
        if len(ref) <= MAX_REF_LEN and lang in priority_langs:
            print(f"\n  [{lang}] {uid}  (CER gap = {diff:.2f})")
            print(f"  REF:  {ref}")
            print(f"  RLVR: {rr['prediction'][:200]}  (CER={rr['cer']:.3f})")
            print(f"  LoRA: {lr['prediction'][:200]}  (CER={lr['cer']:.3f})")
            shown += 1
            if shown >= TOP_EXAMPLES:
                break


def main():
    # Load all files
    data = {}
    for name, path in FILES.items():
        with open(path) as f:
            data[name] = json.load(f)

    # Compare on each dataset
    compare(data["20% RLVR (SL)"], data["100% LoRA (SL)"], "SwitchLingua")
    compare(data["20% RLVR (CF)"], data["100% LoRA (CF)"], "CS-FLEURS read_test")


if __name__ == "__main__":
    main()
