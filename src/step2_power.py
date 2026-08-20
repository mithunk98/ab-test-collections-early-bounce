"""
Step 2 - Power analysis and feasibility.

Answers the question that has to be settled BEFORE randomisation:
given how many early-bounce accounts we actually generate per month,
what size of effect can this experiment realistically detect?

Run:  python src/step2_power.py
"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

import config as C

warnings.filterwarnings("ignore")

POWER_SOLVER = NormalIndPower()


def n_per_arm(p0: float, mde: float, alpha=C.ALPHA, power=C.POWER) -> float:
    """Sample size per arm for a two-proportion test."""
    effect = proportion_effectsize(p0 + mde, p0)
    return POWER_SOLVER.solve_power(
        effect_size=abs(effect), alpha=alpha, power=power,
        ratio=1.0, alternative="two-sided",
    )


def detectable_mde(p0: float, n: float, alpha=C.ALPHA, power=C.POWER) -> float:
    """Smallest absolute lift detectable with n per arm. Inverts the above."""
    lo, hi = 0.0001, 0.50
    for _ in range(60):
        mid = (lo + hi) / 2
        if n_per_arm(p0, mid, alpha, power) > n:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    cohort = pd.read_csv(C.DATA / "cohort_anonymised.csv")
    monthly = len(cohort)
    p0 = C.BASELINE_CURE_RATE

    print("Step 2 - power analysis")
    print(f"Baseline 30-day cure rate (assumed): {p0:.0%}")
    print(f"Observed cohort accrual:            {monthly} accounts/month")
    print(f"Business-case MDE:                  +{C.MDE_ABSOLUTE:.0%} absolute\n")

    # --- Required sample size across a grid of effect sizes ---------------
    rows = []
    for mde in [0.03, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]:
        n = n_per_arm(p0, mde)
        total = int(np.ceil(n) * 2)
        rows.append({
            "mde_abs_pp": f"+{mde*100:.0f}pp",
            "treated_rate": f"{p0+mde:.0%}",
            "n_per_arm": int(np.ceil(n)),
            "n_total": total,
            "months_to_accrue": round(total / monthly, 1),
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    # --- What can we actually detect in a realistic window? ---------------
    print("\nDetectable effect by accrual window (at 80% power):")
    windows = []
    for months in [1, 2, 3, 6, 9, 12]:
        total = monthly * months
        per_arm = total / 2
        mde = detectable_mde(p0, per_arm)
        windows.append({
            "months": months,
            "n_total": total,
            "n_per_arm": int(per_arm),
            "min_detectable_lift": f"+{mde*100:.1f}pp",
            "feasible_for_business_case": "yes" if mde <= C.MDE_ABSOLUTE else "no",
        })
    wdf = pd.DataFrame(windows)
    print(wdf.to_string(index=False))

    # --- Verdict ----------------------------------------------------------
    need = int(np.ceil(n_per_arm(p0, C.MDE_ABSOLUTE)) * 2)
    months_needed = need / monthly
    print("\n" + "=" * 66)
    print("FEASIBILITY VERDICT")
    print("=" * 66)
    print(f"Detecting the +{C.MDE_ABSOLUTE:.0%} business-case effect needs "
          f"{need:,} accounts\n  = {months_needed:.1f} months of accrual "
          f"at {monthly} accounts/month.")
    if months_needed > 6:
        three_mo = detectable_mde(p0, monthly * 3 / 2)
        print("\n  -> NOT FEASIBLE as specified. A test running longer than ~2 "
              "quarters\n     will be overtaken by portfolio and policy drift, "
              "which breaks\n     the comparability the design depends on.")
        print(f"\n  -> In a realistic 3-month window this design can only detect "
              f"+{three_mo*100:.1f}pp\n     or larger. Anything smaller reads as "
              "'no effect' whether or not it\n     is real - the test would be "
              "measuring its own noise floor.")

        # --- The fix: a continuous primary metric -------------------------
        print("\n" + "-" * 66)
        print("REDESIGN - continuous primary metric")
        print("-" * 66)
        print("A binary cure flag throws away most of the information in each")
        print("account. Amount-recovered ratio (Rs collected / Rs due in 30d)")
        print("uses the full distribution, so it needs far fewer accounts.\n")

        from statsmodels.stats.power import TTestIndPower
        tt = TTestIndPower()
        crows = []
        for d in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
            n = tt.solve_power(effect_size=d, alpha=C.ALPHA,
                               power=C.POWER, ratio=1.0, alternative="two-sided")
            total = int(np.ceil(n) * 2)
            crows.append({
                "cohens_d": d,
                "interpretation": ("small" if d < 0.3 else
                                   "small-medium" if d < 0.45 else "medium"),
                "n_per_arm": int(np.ceil(n)),
                "n_total": total,
                "months_to_accrue": round(total / monthly, 1),
            })
        cdf = pd.DataFrame(crows)
        print(cdf.to_string(index=False))
        feasible = cdf[cdf["months_to_accrue"] <= 3.0]
        if not feasible.empty:
            best = feasible.iloc[0]
            print(f"\n  -> A 3-month test CAN detect d = {best['cohens_d']} "
                  f"({best['interpretation']}) on the\n     continuous metric "
                  f"- {best['n_total']} accounts, "
                  f"{best['months_to_accrue']} months. This is the design to run.")
        cdf.to_csv(C.OUTPUTS / "power_continuous_metric.csv", index=False)

        print("\n  -> Recommended: primary = amount_recovered_ratio (continuous),")
        print("     secondary = cured_30d (binary, reported with CI but")
        print("     explicitly underpowered and not used for the go/no-go).")

    C.OUTPUTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(C.OUTPUTS / "power_sample_size.csv", index=False)
    wdf.to_csv(C.OUTPUTS / "power_by_window.csv", index=False)
    print(f"\nWritten: {C.OUTPUTS/'power_sample_size.csv'}, "
          f"{C.OUTPUTS/'power_by_window.csv'}")


if __name__ == "__main__":
    main()
