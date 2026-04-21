# Methodology

## CCA Wiener Ridge (CCA-WR) deconfounding

### Why

Raw embeddings from `gemini-embedding-2-preview` encode everything a
posting says: role, company voice, industry jargon, seniority, office
culture, salary language. For retrieval ("find listings like this one")
that's mostly fine. For *neighborhood discovery* ("which roles share
the same responsibilities?") it's a problem — two ML Engineer postings
at Anthropic end up closer to each other than an ML Engineer posting
at Anthropic is to an ML Engineer posting at Stripe, purely because
Anthropic's boilerplate looks like itself.

The CCA-WR transform is a 768×768 matrix `W` that projects raw
embeddings into a responsibility-focused subspace. Applying
`x_clean = L2(x @ W)` suppresses company-voice and title-wording
signal while preserving what the role actually *is*.

### How it was trained

On 3,580 paired examples drawn from analyzed job listings:

* `role_vec` — the raw embedding (title + description).
* `resp_vec` — a second embedding from only the *responsibilities*
  section of the same listing.

Both share the same target (what someone does on the job), but
`resp_vec` has the company/boilerplate signal naturally stripped.
The training objective: find `W` such that `role_vec @ W` looks as
much like `resp_vec` as possible, in a way that generalizes to held-
out companies.

The three steps:

1. **SVD of cross-covariance.** Compute `C = role_vec.T @ resp_vec`
   (Gram matrix in the paired-sample sense) and take its SVD
   `C = U @ diag(s) @ V.T`. The columns of `U` are directions in the
   role-space that correlate with something in the resp-space; `s`
   gives the strength of each correlation.

2. **Wiener weighting.** Each direction `u_i` gets a weight
   `w_i = s_i / (s_i + λ)`. Strong shared directions pass through
   (w → 1); weak noisy directions are suppressed (w → 0). `λ` is a
   tunable noise floor picked by held-out ablation.

3. **Ridge regression in the weighted space.** Fit a ridge regression
   from role_vec (projected onto the weighted basis) to resp_vec to
   learn the final rotation and scaling.

The composition is distilled into a single 768×768 matrix `W`, saved
as `W` in `data/deconfound_transform.npz`. Application at inference is
one matrix multiply.

### Evaluation

All numbers are on held-out folds from **GroupKFold by company** — no
company appears in both train and test, so the improvements aren't
from memorizing company-specific writing styles.

| metric                                 | raw    | deconfounded | Δ      |
|----------------------------------------|--------|--------------|--------|
| RNP@20 (resp k-NN preserved in role k-NN) | 0.421  | 0.515        | **+22%** |
| RRC (rank correlation of pairwise sims)   | 0.612  | 0.683        | +12%   |
| CI (same-title cross-company cosine)      | 0.543  | 0.607        | +12%   |
| R² (resp_vec reconstruction)              | 0.38   | 0.64         | +68%   |

In plain English: after the transform, positions that share
responsibilities (but not company) are 22% more likely to be among
each other's 20 nearest neighbors, and same-title postings at
different companies are measurably more similar to each other.

## Title normalization

`jobproc.title_normalize.normalize_title(raw) -> str | None`

The neighborhood analysis groups positions by normalized title. Raw
titles are noisy in predictable ways, so the normalizer does a single
pass of regex-based cleanup:

* Strip seniority prefixes: "Senior", "Staff", "Principal", "Lead",
  "Junior", "Sr.", "Founding", "Entry-level".
* Strip trailing location/contract/workstyle:
  - `@ Second Dinner` → dropped
  - `- Remote`, `- Contract`, `- Hybrid` → dropped
  - `- San Francisco, CA` → dropped
  - `(remote)`, `(m/w/d)` → dropped
* Strip trailing Roman-numeral levels: `II`, `III`, `IV`.
* Reject as `None` (don't group):
  - Director / VP / C-suite titles
  - Combo roles ("Designer / Developer", "Sales & Marketing")
  - Internships, recruiter postings, "dream job" marketing
  - Anything under 3 characters
* Smart title case preserving acronyms: `UX`, `ML`, `AI`, `API`, `AWS`,
  `DevOps`, `SaaS`, … (full list in `jobproc/title_normalize.py`).
* `"Events Manager"` → `"Event Manager"` (so plurals don't become
  their own group).

Result: ~3,000 raw titles collapse to ~800 distinct groups in a full
corpus run, tight enough that the top-12 neighbors of any center role
are usually meaningful roles rather than spelling variants.

## Neighborhood selection

For a given center role the algorithm is bottom-up — no a-priori list
of "similar roles", just nearest neighbors in the embedding space:

1. **Find the center** by SQL `ILIKE '%pattern%'` against raw titles,
   then a second filter using `normalize_title` to prevent drift.
   Typical center size: 300-1,500 listings.
2. **Compute the raw centroid.** Mean of center vectors, L2-normalized.
3. **Prefetch candidates.** `ORDER BY role_vec <=> centroid LIMIT N`
   across the full active corpus. N defaults to
   `max(n_neighbors * 5, n_neighbors + center_size + 2000)`.
   Bypasses the HNSW index (see below) so we get the full N, not ~40.
4. **Deconfound both sides** — apply `W`, L2-renormalize.
5. **Re-rank** by cosine to the deconfounded centroid.
6. **Keep top n_neighbors** (default 700).
7. **Group by normalized title.** Groups with ≥6 members make the
   legend; at most 12 groups total to keep the display legible.
8. **Down-sample the center** to at most 200 dots for the 2D layout;
   more than that and t-SNE turns the center into an indistinct blob.

### Why bypass the HNSW index for the prefetch

`pgvector` HNSW indexes are configured with `ef_search` (default 40).
`LIMIT 3000` on a `<=>` query with the index enabled returns
~`ef_search` rows — not 3,000. For retrieval at K=10 this is fine; for
our prefetch it's catastrophic. We `SET enable_indexscan = off` before
the prefetch query, accept the seq-scan cost (a few hundred ms at 90K
rows), and get the full N candidates.

## Layout

PCA(768→50) → t-SNE(50→2D), with `random_state=42` fixed so every run
gives the same layout for the same inputs. Perplexity scales with
position count: `min(150, max(20, n/4))`. The 2D coordinates are
linearly rescaled to [-1, 1] on each axis before rendering.

t-SNE is a **visualization**, not a distance-preserving operation —
relative positions in the 2D plot convey structure but absolute
distances don't correspond to embedding cosine. Groups that are far
apart on the plot *are* far apart in the embedding; groups that touch
on the plot are usually near-neighbors but not always.

## Sanity panels

Each report shows two side panels:

* **Other similar jobs nearby** — top 20 individual listings that are
  in the nearest-N by cosine but whose normalized title didn't make
  it into a named group. Useful for spotting role variants that the
  title normalizer collapsed.
* **Most different jobs in the corpus** — a separate pgvector query
  for the *farthest* listings from the raw centroid, filtered to
  plausible English titles. If the farthest jobs still look similar
  to the center, something is wrong with the embedding space or the
  corpus coverage.

Use these panels as a reality check before drawing conclusions from
the main visualization.
