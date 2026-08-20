"""
Step 1 - Build the eligible cohort from real operational extracts.

Reads the live early-bounce and early-risk dealership reports, applies the
eligibility rules, and writes an ANONYMISED cohort file. No customer name,
phone number or employee email leaves this script.

Run:  python src/step1_cohort.py
"""

import hashlib
import warnings

import pandas as pd

import config as C

warnings.filterwarnings("ignore")

# Columns that must never reach the output file.
PII_COLUMNS = [
    "Customer_Name", "Customer_Phone", "Dealer_Contact_Email",
    "ACM_Name", "ACM_Email", "TSM_Name", "TSM_Email", "SO_Name",
]


def _surrogate(value: str, prefix: str) -> str:
    """Stable pseudonym. Same input always maps to the same id, so the
    cohort can be re-derived and joined back internally, but the mapping
    is not reversible from the published file alone."""
    if pd.isna(value):
        return f"{prefix}_UNKNOWN"
    digest = hashlib.sha256(str(value).strip().lower().encode()).hexdigest()
    return f"{prefix}_{digest[:8].upper()}"


def load_sources() -> pd.DataFrame:
    frames = []
    for path in C.SOURCE_FILES:
        if not path.exists():
            print(f"  [skip] missing: {path.name}")
            continue
        df = pd.read_excel(path)
        df["source_report"] = path.parent.name
        frames.append(df)
        print(f"  [read] {path.name}: {len(df)} rows")
    if not frames:
        raise FileNotFoundError("No source extracts found - check config.SOURCE_FILES")
    return pd.concat(frames, ignore_index=True)


def build_cohort(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # --- Eligibility ------------------------------------------------------
    # Early-stage delinquency only. Later buckets follow a different
    # collections path, so mixing them would dilute the treatment.
    before = len(df)
    df = df[df["EMI_Vintage"].isin(["FEMI", "SEMI", "TEMI"])]
    print(f"  [filter] early-bounce vintage only: {before} -> {len(df)}")

    # Drop accounts with no dealer owner - the treatment is a dealer alert,
    # so these cannot receive it and must not sit in either arm.
    before = len(df)
    df = df[df["Dealer_Master_Name"].notna()]
    print(f"  [filter] dealer-owned accounts: {before} -> {len(df)}")

    # One row per customer - the unit of randomisation.
    before = len(df)
    df = df.drop_duplicates(subset=["Customer_Phone"], keep="first")
    print(f"  [filter] deduplicate to one row per customer: {before} -> {len(df)}")

    # --- Anonymise --------------------------------------------------------
    out = pd.DataFrame({
        "customer_id": df["Customer_Phone"].map(lambda v: _surrogate(v, "CUST")),
        "dealer_id": df["Dealer_Master_Name"].map(lambda v: _surrogate(v, "DLR")),
        "acm_id": df["ACM_Name"].map(lambda v: _surrogate(v, "ACM")),
        "tsm_id": df["TSM_Name"].map(lambda v: _surrogate(v, "TSM")),
        "city": df["City"],
        "emi_vintage": df["EMI_Vintage"],
        "emi_amount": df["EMI_Amount"],
        "total_outstanding": df["Total_Outstanding"],
        "emi_start_date": pd.to_datetime(df["EMI_Start_Date"]),
        "source_report": df["source_report"],
    })

    # City tier by cohort volume - keeps strata from fragmenting into
    # cells of one or two accounts, which would break stratified assignment.
    counts = out["city"].value_counts()
    big = set(counts[counts >= 10].index)
    mid = set(counts[(counts >= 4) & (counts < 10)].index)
    out["city_tier"] = out["city"].map(
        lambda c: "T1" if c in big else ("T2" if c in mid else "T3")
    )

    assert not any(col in out.columns for col in PII_COLUMNS), "PII leaked into cohort"
    return out.reset_index(drop=True)


def main():
    print("Step 1 - building eligible cohort")
    raw = load_sources()
    cohort = build_cohort(raw)

    C.DATA.mkdir(parents=True, exist_ok=True)
    path = C.DATA / "cohort_anonymised.csv"
    cohort.to_csv(path, index=False)

    print(f"\nCohort: {len(cohort)} accounts | "
          f"{cohort['dealer_id'].nunique()} dealers | "
          f"{cohort['city'].nunique()} cities")
    print(f"Exposure: Rs {cohort['total_outstanding'].sum():,.0f} outstanding, "
          f"mean EMI Rs {cohort['emi_amount'].mean():,.0f}")
    print("\nStrata cell sizes:")
    print(cohort.groupby(["emi_vintage", "city_tier"]).size().to_string())
    print(f"\nWritten: {path}")


if __name__ == "__main__":
    main()
