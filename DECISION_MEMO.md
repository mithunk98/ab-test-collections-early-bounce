# Decision Memo — Early-Bounce Dealer Alerting

**To:** Chief Risk Officer · Head of Collections
**From:** Business Analyst, Risk Analytics
**Re:** Proposal to run a controlled test on the dealer early-bounce alert
**Date:** August 2026

---

## Recommendation

**Run a 3-month controlled test at 50/50 allocation on 260 early-bounce accounts, measuring amount recovered rather than cure rate.** Approval needed on one point only: withholding the alert from ~130 accounts for the duration.

---

## Why now

The early-bounce alert currently reaches **100% of eligible accounts**. It has never been tested against a holdout, so we cannot say what it contributes. It costs dealer and field-team attention every week, at ~94 accounts per month.

We are either spending that attention well or spending it for nothing, and at present we have no way to tell. One quarter of controlled measurement settles it permanently.

## What we would measure

| | |
|---|---|
| **Primary** | ₹ recovered ÷ ₹ due, within 30 days of bounce |
| **Secondary** | Share of accounts fully cured in 30 days |
| **Guardrail** | Customer escalations — test stops if these rise more than 2pp |
| **Duration** | ~2.8 months to reach 260 accounts |

## The one design point worth your attention

We originally specified cure rate as the primary metric. **At our volume that test would take 23 months** to detect the +6pp improvement that would make the alert worth its cost — long enough that portfolio and policy drift would invalidate the comparison.

Measuring **amount recovered** instead reaches a conclusion in under three months on the same population, because a partial payment carries information that a yes/no cure flag throws away.

## What we are accepting

- **~130 accounts will not receive the dealer alert for one quarter.** They continue to receive the full standard collections process — telecalling, field visits, notices. Nothing is withheld except this one email.
- **The result will understate the true effect.** Half our dealers will hold accounts in both arms, and a dealer alerted about some customers may act on others. The estimate is therefore a floor, not a point estimate.
- **A negative result is still a win.** If the alert does nothing measurable, we retire it and give the field team back that attention.

## Decision rule, fixed in advance

| Outcome | Action |
|---|---|
| Recovery improves, guardrail holds | Keep the alert; extend to remaining segments |
| No measurable difference | Retire the alert; redeploy the effort |
| Escalations breach +2pp | Stop immediately, regardless of recovery |

The metric, sample size and stopping point are fixed before the test starts. We will not stop early on a good week — doing so is the most common way experiments produce results that fail to replicate.

## Ask

Approval to withhold the dealer alert from a randomised 50% of early-bounce accounts for one quarter, beginning next cycle.
