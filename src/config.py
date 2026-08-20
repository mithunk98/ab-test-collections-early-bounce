"""
Experiment configuration.

Every design parameter lives here so the pre-registration, the power
calculation and the final analysis all read from one source of truth.
Changing a number after randomisation invalidates the test - see
README section "Pre-registration".
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"

# --- Source files (real operational extracts, anonymised on ingest) -------
SOURCE_FILES = [
    Path.home() / "Desktop/Reports/Early_Risk_DealershipLevel_Email"
    / "query_result_2026-07-27T18_02_44.241117733+05_30.xlsx",
    Path.home() / "Desktop/Reports/EarlyBounceDealership_Email(1_2_3)"
    / "femi_semi_temi_bounce_report_2026-07-30T15_48_41.227850034+05_30.xlsx",
]

# --- Hypothesis -----------------------------------------------------------
# H0: dealer-level early-bounce alerting has no effect on 30-day cure rate.
# H1: it changes the 30-day cure rate.
PRIMARY_METRIC = "cured_30d"          # binary: bounced EMI cleared within 30 days
GUARDRAIL_METRIC = "escalation_30d"   # binary: customer complaint / escalation raised
SECONDARY_METRICS = ["days_to_cure", "amount_recovered_ratio"]

# --- Design ---------------------------------------------------------------
ALPHA = 0.05            # two-sided
POWER = 0.80
ALLOCATION = 0.50       # share to treatment
TWO_SIDED = True

# Baseline 30-day cure rate for the early-bounce population.
# PLACEHOLDER - replace with the true rate from Metabase before running.
# Query: cure rate for FEMI/SEMI/TEMI bounced accounts, trailing 6 months.
BASELINE_CURE_RATE = 0.45

# Minimum detectable effect, absolute percentage points.
# Set from the business case, not from what is convenient:
# at ~26k average EMI, +6pp cure on ~350 monthly bounces is the point
# at which the intervention pays for the ops time it consumes.
MDE_ABSOLUTE = 0.06

# Guardrail: we stop the test if escalations rise by more than this.
GUARDRAIL_MAX_INCREASE = 0.02

# --- Stratification -------------------------------------------------------
# Bounce severity is the strongest known predictor of cure, so we force
# balance on it rather than trusting randomisation at this sample size.
STRATA = ["emi_vintage", "city_tier"]

RANDOM_SEED = 20260806   # fixed before randomisation, never re-rolled
