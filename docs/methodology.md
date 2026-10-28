# Fixed H1/H2 methods and evidence limits

## Models

H1 uses six equally weighted members: the original F3 HGB (with TF-IDF),
LightGBM and CatBoost members, each once with a 3-month market level and once
with a 6-month market level. All target log price with absolute-error/L1 losses.
Base/property, description keywords and historical-sale features are used;
only HGB includes the TF-IDF group. List price is excluded.

H2 uses five members: market-adjusted log-price HGB at 3 and 6 months (0.2 each),
HGB targeting log(sale/list) without market adjustment (0.4), and enhanced
market-adjusted Ridge at 3 and 6 months (0.1 each, alpha=3). All include the
list-price feature group. Both ensembles use geometric averaging,
`exp(sum(normalized_weight * log(member_price)))`.

The source configurations and learning code are unchanged by migration.
`features.py`, `history.py`, `market.py` and `text.py` supply feature/state builders.
Intended historical inputs satisfy `date_sold < as_of`; actual days-on-market
and external AVM estimates are excluded. Synthetic tests check useful contracts
but do not establish that every validation/calibration detail is leakage-free.

## Selection and validation

Summer23 trained before June 2023 and predicted June-September; it was the tuning
split. Summer24/25 use the same season/horizon in later years and also informed
selection. Rise21 trained before October 2021 and predicted October-January:
different season and only nine months of training, used as a directional check.

The hedge did not satisfy the original rule of improving all three summers by
at least 0.05 percentage points. It was selected as a robustness judgment based
on mean/worst-case behavior, with small degradations in the flatter summers.
Do not describe it as having passed the original rule, or claim that its exact
advantage has been established by an untouched test. Cross-machine GPU results
vary; attribute comparisons to their actual machines and run types.

The lab subsequently completed the one April-May 2026 confirmation. Each track
had 4,515 homes. H1 MdAPE/MAPE were 7.115%/9.448%; H2 were 4.269%/5.222%.
These are imported recorded results; the local migration only reloads saved
models on unlabeled test inputs and runs synthetic tests. Final model selection
is fixed; no new real training or confirmation is performed during migration.

## Checks still open

The internal early-stopping split uses feature/history/market/text state built
from the outer training set first. Whole-training price-statistic fallbacks,
text target centering and text cross-validation retain the earlier audit limits.
This migration preserves the chosen models rather than treating those checks
as completed or changing the algorithm after confirmation.

Intervals reuse early-stopping residuals, and geometric endpoint blending does
not establish nominal coverage. H1's recorded confirmation coverage was 79.98%
with median width 21.92%; agreement with 80% on this subset is not a guarantee
on future homes. H2 has no complete ensemble interval because its Ridge members
do not provide intervals. Hidden-test sale prices/scores are unavailable locally.

## Provenance and reproduction

Saved models, submission CSVs, reference copies and run records belong to the
same lab generation. File hashes and complete imported H1/H2 metric/spec records
are consolidated in `reports/validation_evidence.json`. The imported completed
confirmation ledger blocks reruns. Original per-home confirmation predictions
and source/data freeze fingerprints were not transferred: do not fabricate them
or reconstruct the official six-month validation from aggregate metrics.

The old F3/F4 release and supplied work log/research scripts are backed up outside
the project. Historical HGB A2/M2 market-adjustment control metrics remain in the
evidence JSON for context.
