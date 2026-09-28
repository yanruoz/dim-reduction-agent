# Data type taxonomy (numeric feature matrices)

Adapted from a general clinical/tabular data-preprocessing taxonomy to this project's narrower
scope: every dataset here is one numeric `(n_samples, n_features)` matrix (the loader refuses
non-numeric columns), reduced by one shared preprocessing block applied identically to every
feature (CLAUDE.md's guard rail against per-method or per-column preprocessing). So unlike a
general tabular taxonomy, there is no per-column role/type assignment here (no id/timestamp/
categorical columns survive the loader to classify) — the classification below is of the whole
matrix's *modality*, used once per dataset to pick the shared preprocessing block.

Name the modality explicitly in `plan.json`'s preprocessing reasons (data-profiling/SKILL.md's
decision tree branches map to these), not just the raw profile numbers on their own — a number
alone ("97.4% sparse") doesn't say why that number implies a particular preprocessing choice for
*this kind* of data; naming the modality is what connects the two.

## Sparse count / omics-like

**Signature:** non-negative, integer- or near-integer-valued, high sparsity (most values zero),
`DATA_DESCRIPTION.md` describes counts, reads, UMIs, or similar (e.g. scRNA-seq, bag-of-words).

**Why it needs different handling than a generic continuous matrix:** raw counts confound
biological/informational signal with per-sample "depth" (total counts per cell, total words per
document); comparing raw counts across samples of different depth is comparing apples to oranges.
The distribution is also heavy-tailed (a handful of highly-expressed features dominate the raw
scale), which would let those few features dominate any distance-based method.

**Typical preprocessing:** `normalize_total` (remove the depth confound) then `log1p` (compress
the tail), in that order; `select_top_variance` if the feature count is large relative to sample
count. Standardizing on top of this is not a default — see data-profiling/SKILL.md's own tested
counter-example, where it flattened the variance spectrum without any labels to show it helped.

## Dense, naturally bounded-range (image/pixel-like)

**Signature:** dense (little to no sparsity), values confined to a fixed, known range regardless
of sample (e.g. 0-255 pixel intensities), or `DATA_DESCRIPTION.md` states a fixed range directly.

**Why it needs different handling:** the bound itself is not informative variation, it's a
storage artifact (an 8-bit encoding, a normalized sensor reading); rescaling to a common range
just removes that artifact so no feature dominates purely because its native encoding happened to
use a larger number range than another feature's.

**Typical preprocessing:** `scale_unit_range`. Not `standardize`: a bounded range already implies
comparable per-feature scale, so standardizing would only re-introduce different effective scales
based on how much of the fixed range that particular feature happens to use in this sample.

## Dense, continuous, unbounded, variable per-feature scale (generic continuous)

**Signature:** dense, no natural fixed bound, and features are not already on comparable scales
(e.g. one feature in the 0-1 range, another in the thousands) — the common case for a generic
numeric table with no special structure described in `DATA_DESCRIPTION.md`.

**Why it needs different handling:** with no shared bound or count structure to lean on, the only
generic way to keep one large-scale feature from dominating a distance-based method by raw
magnitude alone is to put every feature on the same scale directly.

**Typical preprocessing:** `standardize`.

## Wide (features exceed or rival samples), any of the above

**Signature:** `n_features` in the thousands and/or exceeding `n_samples`, on top of any modality
above.

**Why it needs different handling:** most distance-based and O(N²) methods scale in cost and
noise-sensitivity with feature count; with far more features than samples, most of them typically
carry mostly noise for a handful of truly informative directions (as `explained_variance_ratio`
after PCA usually shows directly for these datasets).

**Typical preprocessing:** `select_top_variance` after any modality-specific step above, not as a
replacement for it — reducing feature count doesn't fix a depth confound or a raw-scale mismatch,
it only reduces how much noise a fixed number of features can add.

## Missing values, any modality

Handled first, before modality-specific preprocessing, regardless of which modality above applies
— see data-profiling/SKILL.md step 0. A modality classification assumes complete data; missing
cells break that assumption for the count-based, sparsity, and range checks above until they're
resolved.

## Ambiguous or mixed signal

If a dataset's profile doesn't clearly match one modality above (e.g. sparse but not integer
valued, or a description that doesn't settle it), don't force it into the closest-sounding
category. State in the `reason` which modalities were considered, what in `profile.json` or
`DATA_DESCRIPTION.md` made the call ambiguous, and which choice was made anyway and why (CLAUDE.md
rule 11: a judgment call, not a silent default).
