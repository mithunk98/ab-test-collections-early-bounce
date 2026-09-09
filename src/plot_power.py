"""Power curves for the early-bounce collections A/B test.

Renders the feasibility argument behind the design choice: how long the test must
accrue to reach the target power, under the original binary cure-flag metric versus
the continuous amount-recovered ratio that replaced it.

The two panels do NOT share an x-axis. A percentage-point lift on a cure rate and a
Cohen's d on a continuous ratio are different units, and drawing them against one
axis would imply a correspondence that does not exist. They share a y-axis instead
(accrual months), which is the quantity the decision actually turned on.

Outputs, written to outputs/:
    power_curve_light.png   for light README backgrounds
    power_curve_dark.png    for dark README backgrounds
    power_curve.csv         the plotted values, so the figure is auditable

Usage:
    python src/plot_power.py

Requires matplotlib in addition to the existing requirements.txt dependencies.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize

# --------------------------------------------------------------------------------
# Design parameters
# --------------------------------------------------------------------------------
# Everything that config.py owns is read from config.py, so this figure cannot
# drift away from the numbers step2_power.py ran on. Only the two values that
# exist nowhere else are defined here.

# config.py name -> the name this script uses, where they differ.
CONFIG_ALIASES = {"TARGET_LIFT_PP": "MDE_ABSOLUTE"}

FALLBACKS = {
    "ALPHA": 0.05,           # from config.C.ALPHA
    "POWER": 0.80,           # from config.C.POWER
    "TARGET_LIFT_PP": 0.06,  # from config.MDE_ABSOLUTE
    "BASELINE_CURE_RATE": 0.45,  # from config.BASELINE_CURE_RATE (itself a PLACEHOLDER)
    # Accrual is not a config constant: step2_power.py takes it as len(cohort),
    # i.e. one month of observed accrual. Read the same way below.
    "ACCRUAL_PER_MONTH": 94,
    # Not in config.py - this is the design point step2_power.py lands on as the
    # first continuous-metric row that fits inside a 3-month window.
    "TARGET_D": 0.35,
    # The 6-month figure step2_power.py tests against ("longer than ~2 quarters
    # will be overtaken by portfolio and policy drift").
    "FEASIBILITY_CEILING_MONTHS": 6.0,
}

# Claims made in README.md / DECISION_MEMO.md, re-checked at the end of the run.
# These match outputs/power_sample_size.csv and outputs/power_continuous_metric.csv.
CLAIMED_TOTAL_N = 260
CLAIMED_CONTINUOUS_MONTHS = 2.8
CLAIMED_BINARY_MONTHS = 23.1

# Reference palette, categorical slots 1 and 2, light and dark steps.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "grid": "#e8e7e3",
        "axis": "#c9c8c3",
        "continuous": "#2a78d6",
        "binary": "#eb6834",
        "ceiling": "#8a8984",
        "band_alpha": 0.07,
    },
    "dark": {
        "surface": "#1a1a19",
        "text_primary": "#ffffff",
        "text_secondary": "#c3c2b7",
        "grid": "#302f2c",
        "axis": "#4a4945",
        "continuous": "#3987e5",
        "binary": "#d95926",
        "ceiling": "#8a8984",
        "band_alpha": 0.18,
    },
}


def load_params() -> dict:
    """Pull design parameters from config.py and the cohort file.

    config.py is the pre-registered source of truth, so it wins over every fallback
    here. Accrual is taken the same way step2_power.py takes it - the row count of
    the anonymised cohort, which is one month of observed accrual.
    """
    params = {**FALLBACKS}
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import config
    except ImportError:
        print("  WARNING: config.py not importable - falling back to hard-coded values")
        return params

    taken = []
    for key in params:
        src = CONFIG_ALIASES.get(key, key)
        if hasattr(config, src):
            params[key] = getattr(config, src)
            taken.append(f"{key}" + (f" (as {src})" if src != key else ""))
    if taken:
        print(f"  from config.py: {', '.join(sorted(taken))}")

    # Accrual: same derivation as step2_power.py.
    cohort_path = Path(config.DATA) / "cohort_anonymised.csv"
    if cohort_path.exists():
        with cohort_path.open(encoding="utf-8") as fh:
            rows = sum(1 for _ in fh) - 1  # minus header
        params["ACCRUAL_PER_MONTH"] = rows
        print(f"  from {cohort_path.name}: ACCRUAL_PER_MONTH = {rows}")
    else:
        print(f"  WARNING: {cohort_path.name} missing - accrual falls back to "
              f"{params['ACCRUAL_PER_MONTH']}/month")

    unset = sorted(set(FALLBACKS) - {t.split(" ")[0] for t in taken} - {"ACCRUAL_PER_MONTH"})
    if unset:
        print(f"  not in config.py, using this script's values: {', '.join(unset)}")
    return params


def continuous_n_per_arm(d: np.ndarray, alpha: float, power: float) -> np.ndarray:
    """Sample size per arm for a two-sample t-test. Mirrors step2_power.py's TTestIndPower."""
    solver = TTestIndPower()
    return np.array([
        solver.solve_power(
            effect_size=e, alpha=alpha, power=power, ratio=1.0, alternative="two-sided"
        )
        for e in d
    ])


def binary_n_per_arm(
    lift: np.ndarray, baseline: float, alpha: float, power: float
) -> np.ndarray:
    """Sample size per arm for a two-proportion test. Mirrors step2_power.py's n_per_arm."""
    solver = NormalIndPower()
    return np.array([
        solver.solve_power(
            effect_size=abs(proportion_effectsize(baseline + delta, baseline)),
            alpha=alpha, power=power, ratio=1.0, alternative="two-sided",
        )
        for delta in lift
    ])


def _total_and_months(n_per_arm: float, per_month: int) -> tuple[int, float]:
    """Round exactly as step2_power.py does, so the figure and the CSVs agree."""
    total = int(np.ceil(n_per_arm) * 2)
    return total, round(total / per_month, 1)


def build_curves(p: dict) -> dict:
    """Compute both curves plus the two design points."""
    d_grid = np.linspace(0.15, 0.80, 160)
    lift_grid = np.linspace(0.02, 0.15, 160)
    per_month = p["ACCRUAL_PER_MONTH"]

    cont_n = continuous_n_per_arm(d_grid, p["ALPHA"], p["POWER"])
    bin_n = binary_n_per_arm(lift_grid, p["BASELINE_CURE_RATE"], p["ALPHA"], p["POWER"])

    cont_total, cont_months = _total_and_months(
        continuous_n_per_arm(np.array([p["TARGET_D"]]), p["ALPHA"], p["POWER"])[0], per_month
    )
    bin_total, bin_months = _total_and_months(
        binary_n_per_arm(
            np.array([p["TARGET_LIFT_PP"]]), p["BASELINE_CURE_RATE"], p["ALPHA"], p["POWER"]
        )[0],
        per_month,
    )

    return {
        "d_grid": d_grid,
        "cont_months": cont_n * 2 / per_month,
        "lift_grid": lift_grid,
        "bin_months": bin_n * 2 / per_month,
        "cont_point": (p["TARGET_D"], cont_months, cont_total),
        "bin_point": (p["TARGET_LIFT_PP"], bin_months, bin_total),
    }


def style_axis(ax, t: dict) -> None:
    """Recessive grid and axes; ink stays in text tokens."""
    ax.set_facecolor(t["surface"])
    ax.grid(True, which="major", color=t["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=t["text_secondary"], labelsize=9, length=0)


def render(curves: dict, p: dict, theme: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    t = THEMES[theme]
    ceiling = p["FEASIBILITY_CEILING_MONTHS"]

    fig, (ax_bin, ax_cont) = plt.subplots(
        1, 2, figsize=(11, 4.6), sharey=True, facecolor=t["surface"]
    )

    panels = [
        (
            ax_bin,
            curves["lift_grid"] * 100,
            curves["bin_months"],
            t["binary"],
            "Binary cure flag",
            "Absolute lift in 30-day cure rate (pp)",
            (curves["bin_point"][0] * 100, curves["bin_point"][1], curves["bin_point"][2]),
            "original design",
        ),
        (
            ax_cont,
            curves["d_grid"],
            curves["cont_months"],
            t["continuous"],
            "Continuous recovery ratio",
            "Standardised effect size (Cohen's d)",
            curves["cont_point"],
            "design as run",
        ),
    ]

    for ax, x, y, color, title, xlabel, point, subtitle in panels:
        style_axis(ax, t)

        # Feasibility ceiling: above this line the market drifts out from under the test.
        ax.axhspan(ceiling, 1e4, color=t["ceiling"], alpha=t["band_alpha"], zorder=1, linewidth=0)
        ax.axhline(ceiling, color=t["ceiling"], linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)

        ax.plot(x, y, color=color, linewidth=2.0, zorder=4, solid_capstyle="round")

        px, py, pn = point
        # 2px surface ring on the marker so it reads against the line it sits on.
        ax.plot(
            px, py, "o", markersize=9, color=color,
            markeredgecolor=t["surface"], markeredgewidth=2, zorder=6,
        )

        verdict = "feasible" if py <= ceiling else "not feasible"
        ax.annotate(
            f"{py:.1f} months\n{pn:,.0f} accounts\n{verdict}",
            xy=(px, py),
            xytext=(14, 16),
            textcoords="offset points",
            fontsize=9,
            color=t["text_primary"],
            linespacing=1.45,
            zorder=7,
            bbox=dict(
                boxstyle="round,pad=0.42", facecolor=t["surface"],
                edgecolor=t["grid"], linewidth=0.9,
            ),
        )

        ax.set_title(
            title, fontsize=11.5, color=t["text_primary"],
            fontweight="bold", loc="left", pad=13,
        )
        ax.text(
            0.0, 1.015, subtitle, transform=ax.transAxes,
            fontsize=9, color=t["text_secondary"], ha="left",
        )
        ax.set_xlabel(xlabel, fontsize=9.5, color=t["text_secondary"], labelpad=8)
        ax.set_yscale("log")
        ax.set_ylim(0.7, max(60, curves["bin_months"].max() * 1.15))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

    ax_bin.set_ylabel(
        "Accrual required (months, log scale)", fontsize=9.5,
        color=t["text_secondary"], labelpad=8,
    )
    ax_bin.text(
        curves["lift_grid"][0] * 100, ceiling * 1.13, "feasibility ceiling",
        fontsize=8.5, color=t["text_secondary"], va="bottom",
    )

    fig.suptitle(
        f"Switching the primary metric is what made the test feasible "
        f"({p['POWER']:.0%} power, α={p['ALPHA']})",
        fontsize=12.5, color=t["text_primary"], fontweight="bold", x=0.5, y=1.0,
    )
    fig.text(
        0.5, 0.925,
        "Same intervention, same accrual rate - only the outcome measure differs. "
        "Axes are not comparable; the y-scale is.",
        fontsize=9, color=t["text_secondary"], ha="center",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=200, facecolor=t["surface"], bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def write_csv(curves: dict, out_path: Path) -> None:
    """Emit the plotted values so a reader can check the figure against the numbers."""
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "x_label", "x_value", "accrual_months", "total_n"])
        for x, m in zip(curves["lift_grid"], curves["bin_months"]):
            w.writerow(["binary_cure_flag", "absolute_lift_pp", f"{x * 100:.3f}", f"{m:.4f}", ""])
        for x, m in zip(curves["d_grid"], curves["cont_months"]):
            w.writerow(["continuous_recovery_ratio", "cohens_d", f"{x:.4f}", f"{m:.4f}", ""])
    print(f"  wrote {out_path}")


def reconcile(curves: dict) -> None:
    """Check the computed design points against the numbers claimed in the docs.

    If this warns, the figure and the README disagree and one of them is wrong.
    Fix the constants above rather than the claim, unless the claim is the error.
    """
    print("\nReconciliation against README / DECISION_MEMO claims")
    print("-" * 58)
    # Tolerances are tight on purpose: this script mirrors step2_power.py's solver,
    # effect-size function and rounding, so the numbers should land exactly. A miss
    # means the two have genuinely diverged, which is the thing worth catching.
    checks = [
        ("continuous, accrual months", curves["cont_point"][1], CLAIMED_CONTINUOUS_MONTHS, 0.05),
        ("continuous, total accounts", curves["cont_point"][2], CLAIMED_TOTAL_N, 0.5),
        ("binary, accrual months", curves["bin_point"][1], CLAIMED_BINARY_MONTHS, 0.05),
    ]
    ok = True
    for label, computed, claimed, tol in checks:
        hit = abs(computed - claimed) <= tol
        ok &= hit
        print(
            f"  [{'OK ' if hit else 'OFF'}] {label:<28} "
            f"computed {computed:>8.1f}   claimed {claimed:>8.1f}"
        )

    if not ok:
        print(
            "\n  WARNING: the figure contradicts the write-up.\n"
            "  config.py or the cohort file has changed since README.md and\n"
            "  DECISION_MEMO.md were written. Re-run step2_power.py, then update the\n"
            "  CLAIMED_* constants in this file and the numbers in the prose to match.\n"
            "  Do not publish the figure until this passes - a repo whose selling point\n"
            "  is pre-registration cannot ship a chart that disagrees with its own memo."
        )
    else:
        print("\n  All design points agree with the write-up.")


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    outputs = root / "outputs"
    outputs.mkdir(exist_ok=True)

    print("Loading design parameters")
    p = load_params()

    print("\nComputing power curves")
    curves = build_curves(p)

    print("\nRendering")
    for theme in ("light", "dark"):
        render(curves, p, theme, outputs / f"power_curve_{theme}.png")
    write_csv(curves, outputs / "power_curve.csv")

    reconcile(curves)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
