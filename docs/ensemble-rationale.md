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

### ovos-stock — KEEP (new family: shipped-default reproduction)

**Composition:** `padatious-high → adapt-high → m2v-high → adapt-medium`.

**Why this family is interesting:** every fighter above this line is a curated fusion —
someone chose which engines to pair and in what order because the pairing seemed
promising. None of them is *the actual thing ovos-core ships today*. Without a faithful
reproduction of the real default, the league has no answer to "does any of this fusion
work actually beat what a stock install already does" — every KEEP verdict above is
implicitly compared to nothing.

**What it measures:** the arena's best-effort faithful reproduction of the shipped
`ovos-config` default `intents.pipeline`, restricted to stages the arena can instantiate
outside a full ovos-core skill-service session (no converse/OCP/fallback/stop —
documented per-stage in the fighter's `notes`). This is the true zero-effort baseline:
if a curated fusion cannot beat ovos-stock, its curation added nothing over what ships by
default.

**Source:** `/home/miro/.venvs/ovos/lib/python3.12/site-packages/ovos_config/mycroft.conf`,
`intents.pipeline` key (verbatim order cited in the fighter's `notes`).

### mycroft-classic — KEEP (new family: historical-ordering reproduction)

**Composition:** `adapt-high → padatious-medium`.

**Why this family is interesting:** ovos-stock answers "does fusion beat today's
default"; mycroft-classic answers a different historical question — the owner's
observation that "the classic mycroft pipeline isn't there" either. Early mycroft-core
ran Adapt as the sole primary intent parser and wired Padatious in as a `FallbackSkill`
(mycroft-core PR #939), the *opposite* stage order from both ovos-stock and padapt
(both padatious-before-adapt). Nothing in the league tested whether that later
re-ordering — putting the neural template engine first — was actually an improvement,
or just a design preference that was never benchmarked against the order it replaced.

**What it measures:** whether keyword-first/template-fallback (the order Mycroft shipped
for years) under- or over-performs the modern template-first/keyword-fallback order,
holding the same two engines constant (only the order and tiers differ from padapt).

**Source:** mycroft-core PR #939 ("Implement Padatious support"); reviewer comment
"Make PadatiousService inherit from FallbackSkill to fix new fallback changes" confirms
Padatious ran as a fallback behind Adapt, not as a co-equal parallel matcher.

### trident & cascade-soft — KEEP (new family: confidence mixers)

**Compositions:** `trident` = `padatious-high → jurebes(mlp_shallow)-medium →
nebulento-low`; `cascade-soft` = `linha-fina-high → palavreado-medium → nebulento-low`.

**Why this family is interesting:** every prior fusion (frankenparse included) picks
engines primarily for *paradigm* diversity — one keyword engine, one template engine,
one fuzzy engine — but stacks them at similar or arbitrary confidence tiers. No fighter
in the league deliberately spans the full high/medium/low confidence ladder with a
different *architecture class* at each rung, testing confidence-descending fusion as a
shape in its own right rather than a paradigm-diversity side effect.

**What it measures:** `trident` mixes three architecturally distinct classifiers by
descending strictness — neural-template-with-exact-gate (Padatious), statistical ML
classifier (Jurebes, using `mlp_shallow`, the top-accuracy baseline on the published
`intents-for-eval` en-US template board — see `frontend-static/public/data/
benchmark-intent_template-intents-for-eval-en-US.json`, accuracy 0.8154 vs. 0.8143 for
the runner-up), and fuzzy string matching (Nebulento). `cascade-soft` mixes by a
different axis — matching *looseness* rather than architecture class — trained
classifier (Linha-Fina) → order-independent keyword bag (Palavreado) → fuzzy matcher
(Nebulento). Comparing the two head-to-head separates "does architecture diversity
across tiers help" from "does a strictness gradient across tiers help."

**exact_match convention:** every Jurebes stage added in this pass sets
`exact_match: false` — Jurebes's own internal exact-template short-circuit would
otherwise duplicate whatever exact/near-exact matching the stage ahead of it (Padatious,
Adapt) already provides, the same subset-duplication trap padatioso fell into below.

### m2v-first & knn-first — KEEP (new family: embedding-fronted)

**Compositions:** `m2v-first` = `m2v-high → padatious-medium`; `knn-first` =
`hierarchical-knn-high → linha-fina-medium`.

**Why this family is interesting:** the owner's directive was explicit — "none uses
embeddings" was true of every fighter in the registry before this pass. Every existing
fighter is either keyword-rule matching (Adapt, Palavreado) or a per-intent trained
template classifier (Padatious, Nebulento, Jurebes, Linha-Fina); none leads with
general-purpose sentence-embedding similarity, which generalizes across paraphrases via
vector distance rather than exact template/keyword structure.

**What it measures:** `m2v-first` leads with Model2Vec dense sentence embeddings (single
learned similarity boundary); `knn-first` leads with Hierarchical-KNN (nearest-neighbour
retrieval over a domain-then-intent embedding index) — two structurally different
embedding mechanisms, deliberately paired with *different* backup engines (Padatious vs.
Linha-Fina respectively) so a head-to-head between the pair isolates the embedding-front
difference rather than being confounded by a shared backup.

### kw-slot-palavreado & tmpl-slot-{nebulento,linhafina,jurebes} — KEEP (new family: replacement studies)

**Compositions:** `kw-slot-palavreado` = `palavreado-high → padatious-medium`;
`tmpl-slot-{nebulento,linhafina,jurebes}` = `adapt-high → {X}-medium` for each of three
template/statistical engines. **`mycroft-classic`** (`adapt-high → padatious-medium`,
see above) is the shared baseline both grids compare against — see "1 fighter per
config" below.

**Why this family is interesting:** the owner's directive named specific unmeasured
replacements — "none measures replacing adapt/palavreado padatious/nebulento/linha-fina/
jurebes." Every prior fusion bundles a *specific pairing choice* with a *specific shape
choice* at once, so a win or loss can't be attributed to either variable alone. This
family holds shape and one engine fixed per grid, varying only the other engine — a
controlled A/B (`kw-slot`) and 4-way (`tmpl-slot`) isolation study.

**What it measures:** the `kw-slot` A/B fixes the shape (`<X>-high → padatious-medium`)
and the backup engine (Padatious), varying only which keyword-paradigm engine (Adapt's
ordered grammar vs. Palavreado's order-independent bag) leads — isolating the
keyword-engine choice specifically: `mycroft-classic` (Adapt) vs. `kw-slot-palavreado`
(Palavreado). The `tmpl-slot` 4-way fixes the shape (`adapt-high → <X>-medium`) and the
front engine (Adapt), varying only which template/statistical engine backs it up —
isolating the template-engine choice specifically, across all four template-paradigm
engines in `ENGINE_REGISTRY`: `mycroft-classic` (Padatious) vs.
`tmpl-slot-{nebulento,linhafina,jurebes}` (Nebulento, Linha-Fina, Jurebes with
`mlp_shallow`, `exact_match: false`).

**1 fighter per config — no kw-slot-adapt or tmpl-slot-padatious:** the "keyword engine =
adapt" / "template engine = padatious" cell that both crossed grids need
(`adapt-high → padatious-medium`) is identical, by construction, to `mycroft-classic`.
Rather than register that same config a second and third time under `kw-slot-adapt` and
`tmpl-slot-padatious` — three competitor_ids scoring the identical pipeline, exactly the
redundancy this league rejects on sight — `mycroft-classic` itself fills that cell in
both grids. Its historical framing (see above) and its role as the shared adapt/padatious
baseline are the same config asked two different questions; there is nothing to
duplicate. Each fighter's `notes` cross-reference this explicitly.

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
| ovos-stock | padatious-high, adapt-high, m2v-high, adapt-medium | KEEP | faithful reproduction of the shipped ovos-core default, arena-runnable subset |
| mycroft-classic | adapt-high, padatious-medium | KEEP | historical adapt-first/padatious-fallback order; also the shared baseline for both kw-slot and tmpl-slot replacement grids |
| trident | padatious-high, jurebes(mlp_shallow)-medium, nebulento-low | KEEP | does an architecture-diverse 3-tier confidence cascade beat 2-engine fusions |
| cascade-soft | linha-fina-high, palavreado-medium, nebulento-low | KEEP | does a strictness-descending (not architecture-diverse) 3-tier cascade help |
| m2v-first | m2v-high, padatious-medium | KEEP | does a dense-embedding front door beat a template-first front door |
| knn-first | hierarchical-knn-high, linha-fina-medium | KEEP | does a KNN-retrieval embedding front door differ from a dense-embedding one |
| kw-slot-palavreado | palavreado-high, padatious-medium | KEEP | keyword-engine A/B: mycroft-classic (Adapt) vs Palavreado, backup+shape fixed |
| tmpl-slot-nebulento | adapt-high, nebulento-medium | KEEP | template-engine 4-way: which backup engine recovers most, front+shape fixed |
| tmpl-slot-linhafina | adapt-high, linha-fina-medium | KEEP | template-engine 4-way: which backup engine recovers most, front+shape fixed |
| tmpl-slot-jurebes | adapt-high, jurebes(mlp_shallow)-medium | KEEP | template-engine 4-way: which backup engine recovers most, front+shape fixed |
