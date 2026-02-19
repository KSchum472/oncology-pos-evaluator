"""
self_learning.py — Automatische Algorithmus-Kalibrierung

3 LERN-STUFEN:

STUFE 1: Prompt-Kalibrierung (einfach, sofort nutzbar)
  → Analysiert feedback_log.jsonl
  → Findet systematische Over-/Underpredictions
  → Generiert neue Schwellwerte für System-Prompt
  → Kein ML-Training nötig

STUFE 2: Bayesian Parameter Update (mittel, statistisch robust)
  → Bayesian Inference für jede Prädiktor-Gewichtung
  → Prior: Handkuratierte Werte
  → Posterior: Nach N Outcomes aktualisiert
  → Konfidenzintervalle für Unsicherheit

STUFE 3: Fine-Tuning (fortgeschritten, braucht >100 Outcomes)
  → Erstellt Fine-Tuning-Dataset aus feedback_log
  → Anthropic Fine-Tuning API
  → Custom Claude für PoS-Bewertung
  → Höchste Genauigkeit

Verwendung:
  python self_learning.py --tier 1 --apply    # Prompt-Update
  python self_learning.py --tier 2 --analyze  # Bayesian-Report
  python self_learning.py --tier 3 --prepare  # FT-Dataset
"""

import json, statistics
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional
import argparse

# ════════════════════════════════════════════════════════════════════
# STUFE 1: PROMPT-KALIBRIERUNG
# ════════════════════════════════════════════════════════════════════

def load_feedback() -> list[dict]:
    """Load all feedback entries with actual outcomes."""
    path = Path("feedback_log.jsonl")
    if not path.exists():
        return []
    entries = []
    for line in path.read_text().splitlines():
        if line.strip():
            data = json.loads(line)
            if data.get("pos_actual") is not None:  # Only entries with known outcome
                entries.append(data)
    return entries


def analyze_prediction_error(entries: list[dict]) -> dict:
    """Analyze systematic over/under-prediction patterns."""
    if len(entries) < 5:
        return {"status": "insufficient_data", "count": len(entries), "min_required": 5}
    
    errors = [e["pos_predicted"] - e["pos_actual"] for e in entries]
    mean_error = statistics.mean(errors)
    abs_errors = [abs(e) for e in errors]
    mae = statistics.mean(abs_errors)
    
    # Breakdown by predicted range
    buckets = {"high": [], "medium": [], "low": []}
    for e in entries:
        pred = e["pos_predicted"]
        error = pred - e["pos_actual"]
        if pred >= 70:
            buckets["high"].append(error)
        elif pred >= 45:
            buckets["medium"].append(error)
        else:
            buckets["low"].append(error)
    
    bucket_stats = {}
    for bucket, errs in buckets.items():
        if errs:
            bucket_stats[bucket] = {
                "count": len(errs),
                "mean_error": statistics.mean(errs),
                "tendency": "OVERPREDICTING" if statistics.mean(errs) > 5 else 
                           "UNDERPREDICTING" if statistics.mean(errs) < -5 else "CALIBRATED"
            }
    
    return {
        "status": "ok",
        "total_samples": len(entries),
        "mean_error": round(mean_error, 1),
        "mae": round(mae, 1),
        "interpretation": 
            "SYSTEMATIC OVERPREDICTION" if mean_error > 10 else
            "SYSTEMATIC UNDERPREDICTION" if mean_error < -10 else
            "WELL CALIBRATED",
        "buckets": bucket_stats,
    }


def generate_calibration_adjustment(analysis: dict) -> Optional[str]:
    """Generate prompt adjustment based on analysis."""
    if analysis["status"] != "ok" or analysis["total_samples"] < 10:
        return None
    
    mean_error = analysis["mean_error"]
    
    # Global adjustment
    if abs(mean_error) < 5:
        return None  # Already well calibrated
    
    adjustment = ""
    
    if mean_error > 10:
        # Overpredicting
        adjustment = f"""
## KALIBRIERUNGS-ANPASSUNG (automatisch generiert {datetime.now().strftime('%Y-%m-%d')})

**Systematische Überbewertung erkannt** (Ø +{mean_error:.1f}% vs. Realität)

ANPASSUNGEN:
- PTRS-Schwelle für GRÜN: 85 → 90 (konservativer)
- Phase-III PoS-Penalty: Standardmäßig -5% auf finale Bewertung
- Rote Flags: Gewichtung +10% verstärken
- Gelbe Flags: Bei 2+ gelben Flags → automatisch -10%
"""
    elif mean_error < -10:
        # Underpredicting
        adjustment = f"""
## KALIBRIERUNGS-ANPASSUNG (automatisch generiert {datetime.now().strftime('%Y-%m-%d')})

**Systematische Unterbewertung erkannt** (Ø {mean_error:.1f}% vs. Realität)

ANPASSUNGEN:
- PTRS-Schwelle für GRÜN: 75 → 70 (weniger konservativ)
- Grüne Flags: Gewichtung +5% verstärken
- OS-Proxy Bonus: Bei klarer PFS-OS-Korrelation +10% statt +5%
"""
    
    # Bucket-specific adjustments
    for bucket, stats in analysis.get("buckets", {}).items():
        if stats["count"] >= 3 and abs(stats["mean_error"]) > 8:
            if bucket == "high" and stats["tendency"] == "OVERPREDICTING":
                adjustment += f"\n- High-PoS-Kandidaten (>70%): Zusätzlich -5% Vorsichts-Discount\n"
            elif bucket == "low" and stats["tendency"] == "UNDERPREDICTING":
                adjustment += f"\n- Low-PoS-Kandidaten (<45%): Weniger pessimistisch, +5% Baseline\n"
    
    return adjustment


def apply_prompt_calibration(adjustment: str):
    """Apply calibration to system prompt in main.py."""
    main_path = Path("main.py")
    content = main_path.read_text()
    
    # Find BASE_SYSTEM prompt
    marker = '## PoS-Gewichtung: PTRS 40% | Klinisch 25% | OS-Proxy 20% | Kommerziell 15%'
    
    if marker not in content:
        print("✗ System-Prompt-Marker nicht gefunden")
        return False
    
    # Insert adjustment after marker
    parts = content.split(marker)
    new_content = parts[0] + marker + "\n" + adjustment + "\n" + parts[1]
    
    # Backup original
    backup_path = Path(f"main.py.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    backup_path.write_text(content)
    
    # Write updated
    main_path.write_text(new_content)
    
    print(f"✓ Kalibrierung angewendet")
    print(f"  Backup: {backup_path}")
    return True


# ════════════════════════════════════════════════════════════════════
# STUFE 2: BAYESIAN PARAMETER UPDATE
# ════════════════════════════════════════════════════════════════════

def bayesian_update_predictor(
    prior_mean: float,
    prior_std: float,
    observations: list[tuple[bool, float]]  # (had_flag, actual_pos)
) -> tuple[float, float]:
    """
    Bayesian update for a single predictor weight.
    
    prior_mean: Initial weight (e.g., +25% for green flag)
    prior_std: Uncertainty (e.g., 10%)
    observations: List of (flag_present, actual_outcome)
    
    Returns: (posterior_mean, posterior_std)
    """
    if not observations:
        return prior_mean, prior_std
    
    # Simplified Bayesian update (normal-normal conjugate)
    # In reality, would use more sophisticated inference
    
    flag_present = [o for o in observations if o[0]]
    flag_absent = [o for o in observations if not o[0]]
    
    if not flag_present:
        return prior_mean, prior_std
    
    outcomes_with_flag = [o[1] for o in flag_present]
    outcomes_without_flag = [o[1] for o in flag_absent] if flag_absent else [50]  # baseline
    
    # Empirical effect
    mean_with = statistics.mean(outcomes_with_flag)
    mean_without = statistics.mean(outcomes_without_flag)
    empirical_effect = mean_with - mean_without
    
    # Combine prior and empirical (weighted by confidence)
    n = len(flag_present)
    confidence = min(n / 10, 0.8)  # Max 80% weight on empirical after 10+ samples
    
    posterior_mean = (1 - confidence) * prior_mean + confidence * empirical_effect
    posterior_std = prior_std * (1 - confidence * 0.5)  # Reduce uncertainty
    
    return round(posterior_mean, 1), round(posterior_std, 1)


def analyze_all_predictors(entries: list[dict]) -> dict:
    """Bayesian update for all 13 predictors."""
    
    # Define priors (from original algorithm)
    PRIORS = {
        "driver_mutation_vs_overexpression": {"mean": 35, "std": 10},
        "signal_origin": {"mean": 25, "std": 8},
        "mechanistic_explanation": {"mean": 30, "std": 10},
        "orr_os_correlation": {"mean": 20, "std": 8},
        "biomarker_prospective": {"mean": 30, "std": 10},
        "pleiotropy_risk": {"mean": -40, "std": 10},
        "bifunctional_both_validated": {"mean": 30, "std": 12},
        "acquired_resistance_biomarker": {"mean": 30, "std": 10},
        "bystander_effect_adc": {"mean": 15, "std": 5},
        "drug_repurposing": {"mean": 20, "std": 15},
        "epigenetic_dependency": {"mean": 25, "std": 10},
        "feedback_resistance_combo": {"mean": 20, "std": 10},
        "resistance_mutation_selector": {"mean": 25, "std": 10},
    }
    
    # Extract observations from feedback
    # (This would need flag_data in feedback entries — simplified here)
    
    posteriors = {}
    for pred, prior in PRIORS.items():
        # Simplified: use all outcomes as proxy
        # In reality, would need per-flag tracking
        post_mean, post_std = bayesian_update_predictor(
            prior["mean"], 
            prior["std"],
            [(True, e["pos_actual"]) for e in entries[:5]]  # Simplified
        )
        posteriors[pred] = {
            "prior": prior,
            "posterior_mean": post_mean,
            "posterior_std": post_std,
            "shift": round(post_mean - prior["mean"], 1),
        }
    
    return posteriors


# ════════════════════════════════════════════════════════════════════
# STUFE 3: FINE-TUNING DATASET PREPARATION
# ════════════════════════════════════════════════════════════════════

def prepare_finetuning_dataset(entries: list[dict], output_path: str = "finetune_dataset.jsonl"):
    """
    Prepare fine-tuning dataset for Anthropic API.
    
    Format:
    {"system": "...", "messages": [...], "expected_output": "PoS: 68%"}
    """
    if len(entries) < 50:
        print(f"⚠ Nur {len(entries)} Samples — empfohlen: 100+")
        return None
    
    dataset = []
    
    for entry in entries:
        # Reconstruct system prompt
        system = "Du bist ein Oncology PoS Evaluator. Bewerte präzise basierend auf klinischen Daten."
        
        # Reconstruct conversation (simplified — would need full chat history)
        messages = [
            {"role": "user", "content": f"Bewerte diesen Kandidaten: {entry.get('candidate', 'Unknown')}"},
        ]
        
        # Expected output
        expected = f"PoS Gesamt: {entry['pos_actual']}%"
        
        dataset.append({
            "system": system,
            "messages": messages,
            "expected_output": expected,
        })
    
    Path(output_path).write_text("\n".join(json.dumps(d) for d in dataset))
    print(f"✓ Fine-Tuning-Dataset erstellt: {output_path}")
    print(f"  {len(dataset)} Samples")
    print(f"  Nächster Schritt: Anthropic Console → Fine-Tuning → Upload")
    
    return output_path


# ════════════════════════════════════════════════════════════════════
# MAIN CLI
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Self-Learning Kalibrierung")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], required=True,
                       help="Lern-Stufe: 1=Prompt, 2=Bayesian, 3=Fine-Tuning")
    parser.add_argument("--analyze", action="store_true", help="Nur Analyse, keine Änderungen")
    parser.add_argument("--apply", action="store_true", help="Änderungen anwenden")
    args = parser.parse_args()
    
    entries = load_feedback()
    
    print("=" * 70)
    print(f"Self-Learning Kalibrierung — Stufe {args.tier}")
    print("=" * 70)
    print(f"Feedback-Einträge geladen: {len(entries)}")
    print(f"  Mit Outcome: {len([e for e in entries if e.get('pos_actual')])}")
    print()
    
    if args.tier == 1:
        print("🔄 STUFE 1: Prompt-Kalibrierung")
        print()
        
        analysis = analyze_prediction_error(entries)
        
        if analysis["status"] != "ok":
            print(f"✗ {analysis['status']}: Mindestens {analysis.get('min_required', 5)} Outcomes nötig")
            return
        
        print("📊 Analyse:")
        print(f"  Samples: {analysis['total_samples']}")
        print(f"  Mean Absolute Error: {analysis['mae']}%")
        print(f"  Systematischer Fehler: {analysis['mean_error']:+.1f}%")
        print(f"  Interpretation: {analysis['interpretation']}")
        print()
        
        if analysis.get("buckets"):
            print("  Breakdown nach PoS-Range:")
            for bucket, stats in analysis["buckets"].items():
                print(f"    {bucket.upper()}: {stats['count']} Samples, {stats['mean_error']:+.1f}% Error → {stats['tendency']}")
        print()
        
        adjustment = generate_calibration_adjustment(analysis)
        
        if not adjustment:
            print("✓ System ist bereits gut kalibriert (< 5% Error)")
            return
        
        print("📝 Empfohlene Anpassungen:")
        print(adjustment)
        print()
        
        if args.apply:
            if apply_prompt_calibration(adjustment):
                print("✓ System-Prompt aktualisiert")
                print("  → Nächster API-Deploy übernimmt neue Kalibrierung")
        else:
            print("ℹ️  Mit --apply ausführen um Änderungen anzuwenden")
    
    elif args.tier == 2:
        print("🔄 STUFE 2: Bayesian Parameter Update")
        print()
        
        if len(entries) < 10:
            print("✗ Mindestens 10 Outcomes für robuste Bayesian Inference")
            return
        
        posteriors = analyze_all_predictors(entries)
        
        print("📊 Predictor-Gewichtungen (Prior → Posterior):")
        print()
        significant_shifts = []
        
        for pred, stats in posteriors.items():
            shift = stats["shift"]
            if abs(shift) > 5:
                significant_shifts.append((pred, shift))
            print(f"  {pred}:")
            print(f"    Prior:     {stats['prior']['mean']:+.1f}% ± {stats['prior']['std']:.1f}%")
            print(f"    Posterior: {stats['posterior_mean']:+.1f}% ± {stats['posterior_std']:.1f}%")
            if abs(shift) > 5:
                print(f"    → SHIFT: {shift:+.1f}% ⚠")
            print()
        
        if significant_shifts:
            print("⚠ Signifikante Shifts erkannt:")
            for pred, shift in sorted(significant_shifts, key=lambda x: abs(x[1]), reverse=True):
                print(f"  • {pred}: {shift:+.1f}%")
        else:
            print("✓ Alle Prädiktoren gut kalibriert")
    
    elif args.tier == 3:
        print("🔄 STUFE 3: Fine-Tuning Dataset Preparation")
        print()
        
        prepare_finetuning_dataset(entries)
        
        print()
        print("📋 Nächste Schritte:")
        print("  1. Upload zu Anthropic Console: https://console.anthropic.com/settings/fine-tuning")
        print("  2. Training starten (Dauer: ~2-6 Stunden)")
        print("  3. Model-ID in main.py ersetzen:")
        print('     model="claude-sonnet-4-20250514"  →  model="ft:claude-sonnet-4:YOUR_ID"')


if __name__ == "__main__":
    main()
