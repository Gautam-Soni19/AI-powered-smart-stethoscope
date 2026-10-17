# 06_disease_rules.py
# Convert segment-wise sound-event results (normal/crackles/wheezes/both)
# into disease-level decision: Normal / Asthma / COPD / Pneumonia
#
# Input: JSON produced by 05_predict_segments_and_decide.py
# Example:
# python 06_disease_rules.py --json "...\processed\*_segment_decision.json" --save_json
#
# Notes:
# - This is a rule-based clinical-style mapping (heuristic).
# - It does NOT "diagnose" medically; it gives a best-guess label based on sounds.

import os
import json
import argparse
from typing import Dict, Any, Tuple


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing JSON: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def normalize_counts(counts: Dict[str, Any]) -> Dict[str, int]:
    # Expected keys: normal, crackles, wheezes, both
    out = {}
    for k in ["normal", "crackles", "wheezes", "both"]:
        out[k] = safe_int(counts.get(k, 0), 0)
    return out


def normalize_probs(mean_probs: Dict[str, Any]) -> Dict[str, float]:
    # Expected keys: normal, crackles, wheezes, both
    out = {}
    for k in ["normal", "crackles", "wheezes", "both"]:
        out[k] = safe_float(mean_probs.get(k, 0.0), 0.0)
    return out


def decide_disease(
    segment_counts: Dict[str, int],
    mean_probs: Dict[str, float],
) -> Tuple[str, str, Dict[str, float]]:
    """
    Returns:
      disease_label, reason, scores (disease scores for transparency)
    """

    total = sum(segment_counts.values())
    if total <= 0:
        # If no segments info, fall back to probabilities
        # (rare; but safe)
        total = 1

    # Ratios
    r_normal = segment_counts["normal"] / total
    r_crack = segment_counts["crackles"] / total
    r_wheeze = segment_counts["wheezes"] / total
    r_both = segment_counts["both"] / total

    # Quick helpers: "abnormal"
    abnormal_ratio = 1.0 - r_normal

    # --- Disease heuristic scores ---
    # We use both: counts (stronger signal) + probabilities (softer signal)
    # Crackles -> Pneumonia (common proxy)
    # Wheezes -> Asthma/COPD (proxy)
    # Both -> COPD often mixed (proxy), can also be severe infection; we bias COPD.
    #
    # These weights are tuned for practicality, not medical certainty.
    pN = mean_probs["normal"]
    pC = mean_probs["crackles"]
    pW = mean_probs["wheezes"]
    pB = mean_probs["both"]

    scores = {
        "Normal": 0.0,
        "Pneumonia": 0.0,
        "Asthma": 0.0,
        "COPD": 0.0,
    }

    # Normal score: mostly normal segments + high normal prob
    scores["Normal"] = 0.65 * r_normal + 0.35 * pN

    # Pneumonia score: crackles (and some "both") + crackles prob
    scores["Pneumonia"] = 0.70 * r_crack + 0.20 * r_both + 0.10 * pC

    # Asthma score: wheezes (strong) + wheeze prob (soft)
    scores["Asthma"] = 0.80 * r_wheeze + 0.10 * r_both + 0.10 * pW

    # COPD score: mixture of wheeze+crackles and especially "both"
    scores["COPD"] = 0.35 * r_wheeze + 0.35 * r_crack + 0.25 * r_both + 0.05 * pB

    # --- Rule gates (override) ---
    # These prevent silly outputs in edge cases.

    # Gate 1: If almost all segments normal, say Normal
    if r_normal >= 0.85 and pN >= max(pC, pW, pB):
        return "Normal", f"{segment_counts['normal']}/{total} segments normal and normal prob is highest.", scores

    # Gate 2: Strong crackles dominance -> Pneumonia
    if r_crack >= 0.40 and r_crack >= r_wheeze and r_crack >= r_both:
        return "Pneumonia", f"Crackles dominate ({segment_counts['crackles']}/{total} segments).", scores

    # Gate 3: Strong wheeze dominance -> Asthma (more typical)
    if r_wheeze >= 0.40 and r_wheeze > r_crack and r_wheeze >= r_both:
        return "Asthma", f"Wheezes dominate ({segment_counts['wheezes']}/{total} segments).", scores

    # Gate 4: Many 'both' segments -> COPD (mixed pattern)
    if r_both >= 0.20:
        return "COPD", f"Mixed abnormal segments ('both') present ({segment_counts['both']}/{total}).", scores

    # Gate 5: Mixed wheeze+crackle moderate -> COPD
    if (r_wheeze + r_crack) >= 0.40 and abs(r_wheeze - r_crack) <= 0.20:
        return "COPD", f"Mixed wheezes+crackles pattern (w={r_wheeze:.2f}, c={r_crack:.2f}).", scores

    # Otherwise: pick best score
    disease = max(scores, key=lambda k: scores[k])

    # Small sanity: if abnormal exists but score says Normal, push to best abnormal
    if disease == "Normal" and abnormal_ratio >= 0.25:
        # choose among non-normal
        disease = max(["Asthma", "COPD", "Pneumonia"], key=lambda k: scores[k])

    return disease, "Selected by weighted score + sanity checks.", scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to *_segment_decision.json from file 05")
    ap.add_argument("--save_json", action="store_true", help="Save disease decision JSON next to input JSON")
    args = ap.parse_args()

    data = load_json(args.json)

    # Robustly read fields from your 05 output
    wav = data.get("wav") or data.get("WAV") or data.get("audio_path") or "unknown"
    device = data.get("device") or data.get("Device") or "unknown"

    counts_raw = data.get("segment_pred_counts", {})
    probs_raw = data.get("mean_prob_per_class", {})

    counts = normalize_counts(counts_raw)
    probs = normalize_probs(probs_raw)

    disease, reason, scores = decide_disease(counts, probs)

    print("\n=== Disease Decision (Rule-based) ===")
    print(f"Input JSON: {args.json}")
    print(f"WAV: {wav}")
    print(f"Device: {device}")
    print("\nSegment counts:")
    for k in ["normal", "crackles", "wheezes", "both"]:
        print(f"  {k:<8}: {counts[k]}")

    print("\nMean probabilities per class:")
    for k in ["normal", "crackles", "wheezes", "both"]:
        print(f"  {k:<8}: {probs[k]:.4f}")

    print("\nDisease scores (internal):")
    for k in ["Normal", "Asthma", "COPD", "Pneumonia"]:
        print(f"  {k:<9}: {scores[k]:.4f}")

    print("\n✅ Final Disease Prediction:", disease)
    print("Reason:", reason)

    out = {
        "input_json": args.json,
        "wav": wav,
        "device": device,
        "segment_pred_counts": counts,
        "mean_prob_per_class": probs,
        "disease_scores": scores,
        "final_disease_prediction": disease,
        "reason": reason,
    }

    if args.save_json:
        base_dir = os.path.dirname(os.path.abspath(args.json))
        base_name = os.path.splitext(os.path.basename(args.json))[0]
        out_path = os.path.join(base_dir, f"{base_name}_disease_decision.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\n✅ Saved: {out_path}")


if __name__ == "__main__":
    main()
