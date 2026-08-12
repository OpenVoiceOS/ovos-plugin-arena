# Intent ensemble rationale

Every fighter in `registry/competitors/intent/` (the open `intent` fusion league) stacks
two or more engines from `runner/intent_pipeline.py::ENGINE_REGISTRY` into one pipeline.
A fusion earns its place only if it tests a real hypothesis about complementary engine
strengths. Two patterns are nonsense by construction and get dropped on sight:

1. **Stacking an engine with a subset/duplicate of itself.** Padatious already runs
   `padaos` — a regex/exact-match template engine, the same lineage as Padacioso —
   internally, unless `disable_padaos=true` is set. Verified in
   `ovos_padatious/intent_container.py`:
   - `IntentContainer.__init__` builds `self.padaos = padaos.IntentContainer()` unless
     disabled (`disable_padaos` defaults to `False`, and every fighter here leaves it
     `False`).
   - `IntentContainer.calc_intents()` calls `self.padaos.calc_intents(query)` and assigns
     any perfect match `conf=1.0` — the ceiling confidence, checked before/alongside the
     neural score. So a pipeline that puts Padacioso ahead of or beside Padatious to
     "catch the exact matches" is redundant: Padatious already catches them, at the
     highest possible confidence, for free.
2. **A single plugin stacked at multiple confidence tiers with nothing else in the
   pipeline.** A pipeline of just `plugin-high, plugin-medium` (no other engine) is one
   classifier re-asked with a lower bar — it tests whether loosening a threshold helps,
   not whether any *combination* helps, and doesn't belong in a fusion league.

## Engine paradigms (for reference)

| Engine | Paradigm | Matching strategy |
|---|---|---|
| Padatious | template, neural | learned intent-template similarity + internal `padaos` regex/exact gate (conf=1.0 on perfect match) |
| Padacioso | template, GOFAI | pure regex/exact-match on the same `{slot}` template syntax, no learning |
| Nebulento | template, fuzzy | fuzzy string matching over templates, tolerant of typos/ASR noise |
| Adapt | keyword, GOFAI | `IntentBuilder` rule grammar — ordered require/optional/exclude vocab, strict structure |
| Palavreado | keyword, GOFAI | bag-of-keywords, order-independent, no grammar |

## Verdicts

### frankenparse — KEEP (slimmed)

**Composition (after this pass):** `padatious-high → adapt-high → palavreado-medium →
nebulento-medium → adapt-low` — one engine per stage, four distinct paradigms
(neural+exact template, keyword-rule, keyword-bag, fuzzy-template), no paradigm repeated.

**Hypothesis:** does a maximal-diversity cascade — one representative engine per paradigm,
each given exactly one tier — outperform the curated 2-3 engine pairs below it in the
league? This is a real, distinct question from the pairwise fighters: it's the "does more
diversity keep helping past 2 engines, or does it just add latency/false-positive risk"
probe.

**Change made:** the original 7-stage config (`padacioso-high → adapt-high →
padatious-high → palavreado-medium → padatious-medium → nebulento-medium → adapt-low`) had
both nonsense patterns at once:
- `padacioso-high` duplicated Padatious's own internal `padaos` exact-match gate (verified,
  see above) — dropped.
- `padatious-high` and `padatious-medium` were the same trained classifier asked twice at
  different thresholds, with no other stage between them that changes what Padatious itself
  would do on a retry — not an independent paradigm, not separately justified. Slimmed to a
  single `padatious-high` stage.

**Expected strengths:** best plausible recall across paradigm-diverse utterances; a solid
upper bound for "what if you just run everything."

**Expected downsides:** highest latency and instantiation cost in the league (4 engines
vs. 2 for every other fighter); the adapt-low last resort raises false-positive risk over
the pairwise fighters, since keyword rules at a low bar are the loosest gate in the roster.


**Composition:** `padatious-high → adapt-high → adapt-medium`.

**Hypothesis:** does the shipped OVOS default order — neural template first, keyword rules
as a two-tier fallback — beat either engine run alone? Template similarity (learned) and
keyword-rule matching (explicit grammar) are genuinely different paradigms, so this tests
real complementary coverage, not duplication. It also doubles as the arena's baseline for
"how much does fusion buy you over the stock config."

**Expected strengths:** mirrors real-world OVOS behavior; should do well on utterances that
are grammatically clean (Adapt) or template-typical (Padatious) even if not both.

**Expected downsides:** neither engine handles fuzzy/noisy ASR output well, so this pair is
expected to underperform nebulapt/nebulatious specifically on noisy input.

### nebulatious — KEEP

**Composition:** `padatious-high → nebulento-medium → padatious-medium`.

**Hypothesis:** does a neural template classifier at high confidence, backed by a fuzzy
template matcher tuned for typos/ASR noise, beat either alone — and does retrying Padatious
at a looser gate afterward recover cases the fuzzy stage still missed? Padatious's internal
`padaos` exact-match gate already covers regex-perfect input, so this pair specifically
probes the neural-vs-fuzzy paraphrase/noise gap, not exact-match duplication (unlike
padatioso, see below).

**Note on the trailing `padatious-medium` stage:** this is the same "same engine, second
tier" pattern flagged as suspect elsewhere, but it is not the *only* content of the
pipeline — Nebulento sits between the two Padatious tiers and is expected to catch most of
what Padatious's retry would otherwise be asked to. Kept because the fighter's hypothesis is
adapt-medium convention rather than adding a second independent claim.

**Expected strengths:** best-in-league on noisy/typo'd input among the template-only
fighters.

**Expected downsides:** no keyword-paradigm coverage at all — grammatically explicit
commands that neither template engine's training data resembles are not this pipeline's
strength.

### nebulapt — KEEP

**Composition:** `adapt-high → nebulento-medium → adapt-medium`.

**Hypothesis:** does a strict keyword-rule engine (low false-positive, brittle to phrasing
and typos) at high confidence, backstopped by a fuzzy matcher tolerant of ASR noise, beat
either alone? Same shape as nebulatious but for the keyword paradigm instead of template —
Adapt and Nebulento are different strategies (strict rule vs. fuzzy), so this is a real
precision/recall complementarity test, not a duplicate of nebulatious.

**Expected strengths:** best-in-league on noisy/typo'd input among keyword-only fighters;
complements nebulatious as the keyword-side analogue.

**Expected downsides:** no template-paradigm coverage; utterances that don't map cleanly to
Adapt's vocab/grammar and aren't a near-miss of a trained phrase (Nebulento's sweet spot)
fall through.


**Composition:** `adapt-high → palavreado-high → adapt-medium → palavreado-medium`.

**Hypothesis:** within the keyword paradigm, does a strict rule engine (Adapt: ordered
require/optional grammar) combined with a bag-of-keywords engine (Palavreado:
order-independent, no grammar) cover more utterances than either alone? These are genuinely
different keyword-matching strategies, not the same engine at two tiers — this is the
keyword league's pooled-upper-bound probe.

**Expected strengths:** best keyword-only coverage in the league; catches both
grammar-conforming and loosely-phrased keyword utterances.

**Expected downsides:** no template or fuzzy coverage — paraphrases and noisy ASR output
that don't hit a keyword/vocab match will fall through both engines identically, since
neither does any similarity scoring.

### padatioso — DROP

**Composition (removed):** `padacioso-high → padatious-medium`.

**Why it fails justification:** this is exactly the owner's example of engine-subset
duplication. Padatious already runs `padaos` — the same regex/exact-match template engine
Padacioso wraps — as an internal first-class matcher (`IntentContainer.padaos`, enabled by
default), and any perfect match there is assigned `conf=1.0`, the ceiling score, ahead of
or alongside the neural score. Verified directly in
`ovos_padatious/intent_container.py::IntentContainer.calc_intents()`
(`/home/miro/.venvs/ovos/lib/python3.12/site-packages/ovos_padatious/intent_container.py`,
lines ~280-303): `self.padaos.calc_intents(query)` results are folded into the return set
with `conf=1.0` before the container returns. Putting Padacioso in front of Padatious in a
pipeline therefore tests nothing Padatious wasn't already doing internally — it's not a
paradigm mix, it's the same exact-match gate invoked twice under two different plugin
names. **Removed** (`git rm registry/competitors/intent/padatioso.json`).

## Summary

| Fighter | Composition | Verdict | Hypothesis (one line) |
|---|---|---|---|
| frankenparse | padatious-high, adapt-high, palavreado-medium, nebulento-medium, adapt-low | KEEP (slimmed) | does one-stage-per-paradigm max diversity beat curated pairs |
| nebulatious | padatious-high, nebulento-medium, padatious-medium | KEEP | does neural template + fuzzy template beat either alone on paraphrase/noise |
| nebulapt | adapt-high, nebulento-medium, adapt-medium | KEEP | does strict keyword rules + fuzzy keyword matching beat either alone on noise |
| padatioso | padacioso-high, padatious-medium | **DROP** | none — Padatious already runs padaos internally with conf=1.0 on perfect match; padacioso tests nothing padatious doesn't already do |
