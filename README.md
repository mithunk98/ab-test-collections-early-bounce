# Does Early-Bounce Alerting Actually Work? An Experiment Design

**Testing a live collections intervention that was shipped to 100% of accounts with no control group.**

---

## The problem

I built an automation that emails each dealership partner a list of their customers who have just bounced an EMI (1st, 2nd or 3rd instalment), with the area and territory managers auto-CC'd. It runs on schedule and covers every eligible account.

That is the problem. **It went to 100% of the population from day one, so there is no control group and no way to know whether it works.** The alert consumes dealer and field-team attention every week. Nobody can say what it buys.

This project designs the experiment that would answer that, and builds the analysis pipeline to read it out.

---

## Design summary

| | |
|---|---|
| **Hypothesis** | Dealer-level early-bounce alerting changes 30-day repayment behaviour |
| **Unit of randomisation** | Customer account |
| **Arms** | Control (no dealer alert) / Treatment (alert as currently built), 50/50 |
| **Primary metric** | `amount_recovered_ratio` — ₹ collected ÷ ₹ due within 30 days |
| **Secondary** | `cured_30d` — bounced EMI fully cleared within 30 days |
| **Guardrail** | `escalation_30d` — customer complaint or escalation raised; stop if +2pp |
| **Stratified on** | Bounce vintage (FEMI/SEMI/TEMI) × city tier |
| **α / power** | 0.05 two-sided / 0.80 |
| **Pre-registered n** | 260 accounts (~2.8 months of accrual) |

Population as built from live extracts: **94 accounts/month, 24 dealers, 15 cities, mean EMI ₹13,224.**

---

## The three findings that shaped the design

### 1. The obvious design is infeasible, and the power analysis is what proved it

The natural primary metric is cure rate — did they pay or not. At a 45% baseline, detecting the **+6pp lift that makes the intervention worth its ops cost** requires:

| Effect to detect | Accounts needed | Months to accrue |
|---|---|---|
| +3pp | 8,676 | 92.3 |
| +6pp ← business case | **2,176** | **23.1** |
| +10pp | 784 | 8.3 |
| +20pp | 192 | 2.0 |

**23 months.** A test that long is meaningless — portfolio mix, collections policy and the macro environment all drift underneath it, destroying the comparability the design rests on.

Run in a realistic 3-month window, the binary design can only detect **+16.6pp or larger**. An effect smaller than that reads as "no effect" whether or not it is real. The test would be measuring its own noise floor.

### 2. Switching to a continuous metric rescues it

A binary cure flag discards almost everything each account knows. A customer who paid 90% of what was due and one who paid nothing both score zero. Amount-recovered ratio uses the whole distribution:

| Effect size (Cohen's d) | Accounts needed | Months |
|---|---|---|
| d = 0.30 (small-medium) | 352 | 3.7 |
| **d = 0.35** | **260** | **2.8** |
| d = 0.50 (medium) | 128 | 1.4 |

Same population, same window, a test that can actually conclude something. Cure rate stays in the readout as a secondary — reported with confidence intervals, explicitly flagged as underpowered, and **not** used for the go/no-go.

*This is the whole point of running power analysis before the test rather than after: it changed the design instead of explaining a failure.*

### 3. The treatment is dealer-level but the randomisation is customer-level

The alert emails a **dealer** a list of **customers**. Randomising customers means 12 of 24 dealers hold accounts in both arms — and **81% of control accounts sit with a dealer who is receiving alerts about their other customers.**

A dealer told "three of your customers just bounced" may well chase all of theirs. That spillover pushes the measured effect toward zero.

Two ways out:

- **(a) Randomise at dealer level.** Removes spillover completely. But with 24 clusters the effective sample collapses and power gets far worse than the customer-level design it was meant to fix. *Rejected.*
- **(b) Keep the customer-level draw, and pre-register the estimate as an intent-to-treat lower bound**, reporting the contamination rate beside it. *Chosen.*

The consequence is stated up front rather than discovered later: **a null result here cannot be read as "the alert does nothing."** It can only be read as "the alert does not do enough to survive 81% contamination."

---

## Balance and rerandomisation

Stratifying on vintage and city does not constrain EMI size. The first draw came back imbalanced on it — SMD 0.25, well past the 0.10 convention.

Because no outcome data existed yet and the acceptance criterion was fixed in advance, rerandomisation (Morgan & Rubin) is legitimate: redraw until every covariate clears |SMD| < 0.10. Accepted on draw 8.

| Covariate | Treatment | Control | SMD |
|---|---|---|---|
| EMI amount | ₹13,193 | ₹13,255 | −0.020 |
| Total outstanding | ₹10,190 | ₹9,894 | +0.029 |

Balance is judged on standardised mean difference, not a p-value. A balance t-test conflates imbalance with sample size and will happily call a badly skewed small sample "balanced".

---

## Analysis plan (pre-registered)

| Metric | Test |
|---|---|
| Primary | Welch t-test (unequal variance not assumed away), plus OLS with HC3 robust SEs adjusting for strata and EMI size for precision |
| Secondary | Two-proportion z-test with Wilson intervals — Wilson because normal-approximation intervals misbehave badly at these rates and sample sizes |
| Guardrail | One-sided check against the +2pp escalation tolerance |
| Segments | By vintage, Benjamini-Hochberg corrected, declared exploratory |

**Fixed before any data is seen:** the metric, the sample size, the stopping point, and the decision rule. No peeking and stopping early on a good day — that inflates the false-positive rate far above the nominal 5%.

---

## Current status and honest readout

The controlled test **has not been run**. `step4_analyze.py` simulates outcomes under a stated true effect (d = 0.30) so the pipeline is complete and verifiable end to end. Every simulated number is labelled as such. Drop real outcomes into `data/outcomes.csv` and the same script reads them instead.

On the dry run at the pre-registered n = 260:

```
PRIMARY - amount recovered ratio (30d)
  treatment 0.6289   control 0.5535
  difference +0.0754   95% CI [-0.0004, +0.1512]
  p = 0.0513          Cohen's d = +0.243
  -> not significant

GUARDRAIL  treatment 5.3%  control 5.4%  -> within tolerance

DECISION: INCONCLUSIVE - not the same as 'no effect'.
```

**p = 0.0513.** The interval brushes zero. The disciplined answer is that the pre-registered threshold was not met, so the pre-registered decision is "keep accruing" — not "declare victory because it nearly cleared." The direction is consistent and the guardrail is clean, which justifies continuing the test; it does not justify shipping on the strength of a near miss.

---

## Privacy

Source extracts contain real customer names, phone numbers and employee email addresses. `step1_cohort.py` strips every one of them on ingest and replaces the join keys with salted SHA-256 surrogates, asserting before write that no PII column survives. Only the anonymised cohort is ever written to disk here.

---

## Running it

```bash
pip install pandas numpy scipy statsmodels openpyxl
python src/step1_cohort.py      # build + anonymise the eligible cohort
python src/step2_power.py       # power, feasibility, the redesign
python src/step3_randomize.py   # stratified assignment, balance, contamination
python src/step4_analyze.py     # full readout and decision
```

| File | Role |
|---|---|
| `src/config.py` | Every design parameter — one source of truth |
| `src/step1_cohort.py` | Eligibility rules and anonymisation |
| `src/step2_power.py` | Sample size, MDE curves, feasibility verdict |
| `src/step3_randomize.py` | Stratified randomisation, rerandomisation, diagnostics |
| `src/step4_analyze.py` | Pre-registered analysis and decision rule |
| `DECISION_MEMO.md` | The one-page version for the business |
