#!/usr/bin/env python3
"""
Generate data for Figure: Failure mode rates across the RLVR ablation chain.
Measures translation rate and wrong-script rate on non-Latin-script samples.
Models correspond to the 10% data ablation in Table 1.
"""

import json
import unicodedata
from pathlib import Path

# ── Eval files (SwitchLingua + CS-FLEURS, seed 42) ──────────────────────────
EVAL_DIR = Path("/home/ubuntu/Qwen2-Audio/r1-aqa-main/outputs/eval_results")

SWITCHLINGUA = {
    "Base":       EVAL_DIR / "eval_switchlingua_csfleurs_xtts_train_format_n100_e4_all_n500000_20260219_170200.json",
    "CER":        EVAL_DIR / "eval_switchlingua_csfleurs_xtts_train_cer_n2310_e8_novad_s42_n999999_20260221_103742.json",
    "+refine":    EVAL_DIR / "eval_switchlingua_csfleurs_xtts_train_cer_n2310_e4_twostep_novad_s42_n999999_20260222_154612.json",
    "+ref+rew":   EVAL_DIR / "eval_switchlingua_csfleurs_xtts_train_cgpr_plus_n2310_e4_twostep_novad_s42_n999999_20260223_203155.json",
}

CSFLEURS = {
    "Base":       EVAL_DIR / "eval_csfleurs_csfleurs_xtts_train_format_n100_e4_read_test_n500000_20260219_173505.json",
    "CER":        EVAL_DIR / "eval_csfleurs_csfleurs_xtts_train_cer_n2310_e8_novad_s42_read_test_n999999_20260221_121932.json",
    "+refine":    EVAL_DIR / "eval_csfleurs_csfleurs_xtts_train_cer_n2310_e4_twostep_novad_s42_read_test_n999999_20260222_122629.json",
    "+ref+rew":   EVAL_DIR / "eval_csfleurs_csfleurs_xtts_train_cgpr_plus_n2310_e4_twostep_novad_s42_read_test_n999999_20260223_195737.json",
}

# ── Non-Latin languages per dataset ──────────────────────────────────────────
SL_NON_LATIN = {"Arabic", "Japanese", "Korean", "Mandarin", "Russian"}
CF_NON_LATIN = {"ara-eng", "jpn-eng", "kor-eng", "cmn-eng", "rus-eng"}

# ── Expected scripts per language ────────────────────────────────────────────
LANG_SCRIPTS = {
    "Arabic":   {"ARABIC"},
    "Japanese":  {"CJK", "HIRAGANA", "KATAKANA"},
    "Korean":    {"HANGUL"},
    "Mandarin":  {"CJK"},
    "Russian":   {"CYRILLIC"},
    "ara-eng":   {"ARABIC"},
    "jpn-eng":   {"CJK", "HIRAGANA", "KATAKANA"},
    "kor-eng":   {"HANGUL"},
    "cmn-eng":   {"CJK"},
    "rus-eng":   {"CYRILLIC"},
}


def get_scripts(text: str) -> set[str]:
    scripts = set()
    for c in text:
        if c.isspace() or unicodedata.category(c).startswith("P") or c.isdigit():
            continue
        try:
            scripts.add(unicodedata.name(c, "").split()[0])
        except Exception:
            pass
    return scripts


def analyze(results: list[dict], non_latin_langs: set[str]) -> dict:
    non_latin = [r for r in results
                 if r.get("language", r.get("language_pair", "")) in non_latin_langs]
    n = len(non_latin)
    translated = 0
    wrong_script = 0

    for r in non_latin:
        lang = r.get("language", r.get("language_pair", ""))
        ref_scripts = get_scripts(r["reference"])
        pred_scripts = get_scripts(r["prediction"])

        has_nonlatin_ref = bool(ref_scripts - {"LATIN"})
        has_nonlatin_pred = bool(pred_scripts - {"LATIN"})
        has_latin_pred = "LATIN" in pred_scripts

        # Translation: ref has non-Latin, prediction is all Latin
        if has_nonlatin_ref and not has_nonlatin_pred and has_latin_pred:
            translated += 1

        # Wrong script: characters from an unexpected writing system
        expected = LANG_SCRIPTS.get(lang, set()) | {"LATIN"}
        unexpected = pred_scripts - expected - {"FULLWIDTH", "DIGIT"}
        if unexpected:
            wrong_script += 1

    return {
        "n": n,
        "translated": translated,
        "translated_pct": 100 * translated / n if n else 0,
        "wrong_script": wrong_script,
        "wrong_script_pct": 100 * wrong_script / n if n else 0,
    }


def main():
    print("=" * 72)
    print("Figure data: Failure mode rates across ablation chain")
    print("=" * 72)

    for dataset_name, files, non_latin in [
        ("SwitchLingua", SWITCHLINGUA, SL_NON_LATIN),
        ("CS-FLEURS read_test", CSFLEURS, CF_NON_LATIN),
    ]:
        print(f"\n── {dataset_name} ──")
        print(f"{'Model':15s} {'n':>6s} {'Transl%':>8s} {'Script%':>8s}")
        print("-" * 40)
        for model_name, path in files.items():
            with open(path) as f:
                data = json.load(f)
            stats = analyze(data["results"], non_latin)
            print(f"{model_name:15s} {stats['n']:6d} "
                  f"{stats['translated_pct']:7.1f}% "
                  f"{stats['wrong_script_pct']:7.1f}%")

    # ── pgfplots coordinates (copy-paste into LaTeX) ─────────────────────
    print("\n" + "=" * 72)
    print("pgfplots coordinates for LaTeX figure")
    print("=" * 72)

    for dataset_name, files, non_latin, label in [
        ("SwitchLingua", SWITCHLINGUA, SL_NON_LATIN, "SL"),
        ("CS-FLEURS",    CSFLEURS,    CF_NON_LATIN,  "CF"),
    ]:
        trans_coords = []
        script_coords = []
        for model_name, path in files.items():
            with open(path) as f:
                data = json.load(f)
            stats = analyze(data["results"], non_latin)
            coord_name = "{" + model_name + "}"
            trans_coords.append(f"({coord_name},{stats['translated_pct']:.1f})")
            script_coords.append(f"({coord_name},{stats['wrong_script_pct']:.1f})")

        print(f"\n% Transl. ({label})")
        print("\\addplot coordinates {" + " ".join(trans_coords) + "};")
        print(f"% Script ({label})")
        print("\\addplot coordinates {" + " ".join(script_coords) + "};")


if __name__ == "__main__":
    main()
