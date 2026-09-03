#!/usr/bin/env python3
"""Regenerate the OVOS golden-utterances evaluation dataset.

Source of truth: every non-archived ``<org>/<repo-filter>*`` GitHub repo's
default branch, under ``test/end2end/golden_utterances*.jsonl``. Rows are
normalized into the schema published at
``OpenVoiceOS/ovos-golden-utterances-bench`` (one JSONL file per language,
``<lang>/test.jsonl``) and, with ``--publish``, uploaded there via
``huggingface_hub``.

This reproduces the pipeline used to build the live dataset (fetch -> build
-> publish), consolidated into one script so the dataset can be rebuilt
whenever the skill fleet's ``golden_utterances*.jsonl`` suites gain new rows.

Usage::

    python3 scripts/build_golden_utterances.py --out ./golden-utterances-build
    python3 scripts/build_golden_utterances.py --publish

Requires a GitHub token with public read access (``GITHUB_TOKEN`` or
``GH_TOKEN``) for the repo-listing and tree-listing API calls, and an
ambient Hugging Face token (``HF_TOKEN`` or ``huggingface-cli login``) for
``--publish``.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

import requests

DEFAULT_ORG = "OpenVoiceOS"
DEFAULT_REPO_FILTER = "ovos-skill-"
DEFAULT_HF_REPO_ID = "OpenVoiceOS/ovos-golden-utterances-bench"
DROP_GUARD_FRACTION = 0.20

LOCALE_RE = re.compile(r"golden_utterances_([A-Za-z]{2,3}-[A-Za-z]{2,3})\.jsonl$")
LANG_RE = re.compile(r"^[a-z]{2}-[A-Z]{2}$")

GITHUB_API = "https://api.github.com"


def normalize_locale(loc: str) -> str:
    """Normalize a locale string to proper BCP-47 casing (``en-us`` -> ``en-US``)."""
    parts = loc.split("-")
    if len(parts) == 2:
        return f"{parts[0].lower()}-{parts[1].upper()}"
    return loc


def normalize_row(row: dict, repo: str, path: str, file_locale: str | None) -> dict | None:
    """Normalize one raw ``golden_utterances*.jsonl`` row into the published schema.

    Returns ``None`` if the row is missing a required field (``skill_id``,
    ``utterance``, or ``intent_label``), e.g. the dialog-shaped rows some
    suites (like ``fallback-unknown``) ship alongside intent rows.
    """
    skill_id = row.get("skill_id")
    utterance = row.get("utterance")
    intent_label = row.get("intent_label")

    if not skill_id or not utterance or not intent_label:
        return None

    lang = row.get("lang") or file_locale or "en-US"
    lang = normalize_locale(lang)

    stripped_label = intent_label
    if isinstance(intent_label, str) and intent_label.endswith(".intent"):
        stripped_label = intent_label[: -len(".intent")]

    expected_intent = f"{skill_id}:{stripped_label}"

    return {
        "utterance": utterance,
        "expected_intent": expected_intent,
        "lang": lang,
        "skill_id": skill_id,
        "source_repo": repo,
        "source_file": path,
        "intent_label_original": intent_label,
        "intent_type": row.get("intent_type"),
        "intent_method": row.get("intent_method"),
        "needs_manual": bool(row.get("needs_manual", False)),
        "machine_generated": row.get("machine_generated", None),
        "required_vocab": row.get("required_vocab"),
        "expected_messages": row.get("expected_messages"),
    }


def file_locale_from_path(path: str) -> str | None:
    match = LOCALE_RE.search(path)
    if match:
        return normalize_locale(match.group(1))
    return None


# --------------------------------------------------------------------------
# Fetch: enumerate org repos and pull golden_utterances*.jsonl files
# --------------------------------------------------------------------------

def _gh_session() -> requests.Session:
    session = requests.Session()
    session.headers["Accept"] = "application/vnd.github+json"
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    return session


def list_org_repos(org: str, repo_filter: str, session: requests.Session) -> list[str]:
    """Return non-archived repo names in ``org`` whose name starts with ``repo_filter``."""
    repos = []
    page = 1
    while True:
        r = session.get(
            f"{GITHUB_API}/orgs/{org}/repos",
            params={"per_page": 100, "page": page, "type": "public"},
            timeout=60,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        for repo in batch:
            name = repo["name"]
            if repo.get("archived"):
                continue
            if name.startswith(repo_filter):
                repos.append(name)
        page += 1
    return sorted(repos)


def fetch_golden_files(org: str, repos: list[str], session: requests.Session):
    """Fetch every ``test/end2end/golden_utterances*.jsonl`` file for each repo.

    Returns ``(files, no_data)`` where ``files`` is a list of
    ``(repo, path, text)`` and ``no_data`` is a list of ``(repo, reason)``.
    """
    files = []
    no_data = []
    for repo in repos:
        full = f"{org}/{repo}"
        r = session.get(f"{GITHUB_API}/repos/{full}", timeout=60)
        if r.status_code != 200:
            no_data.append((repo, f"repo fetch failed ({r.status_code})"))
            continue
        default_branch = r.json().get("default_branch", "main")

        r = session.get(
            f"{GITHUB_API}/repos/{full}/git/trees/{default_branch}",
            params={"recursive": 1},
            timeout=60,
        )
        if r.status_code != 200:
            no_data.append((repo, f"tree fetch failed ({r.status_code})"))
            continue
        tree = r.json()
        paths = [
            t["path"]
            for t in tree.get("tree", [])
            if t.get("type") == "blob"
            and t["path"].startswith("test/end2end/golden_utterances")
            and t["path"].endswith(".jsonl")
        ]
        if not paths:
            no_data.append((repo, "no golden_utterances files"))
            continue

        for path in paths:
            raw_url = f"https://raw.githubusercontent.com/{full}/{default_branch}/{path}"
            rr = requests.get(raw_url, timeout=60)
            if rr.status_code != 200 or not rr.text.strip():
                no_data.append((repo, f"fetch failed for {path}"))
                continue
            files.append((repo, path, rr.text))
    return files, no_data


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

def build_dataset(files):
    """Normalize fetched files into per-language row lists plus stats."""
    out_rows = collections.defaultdict(list)
    skipped = []
    skills = set()
    intents = set()

    for repo, path, text in files:
        file_locale = file_locale_from_path(path)
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_row = json.loads(line)
            except Exception:
                skipped.append((repo, path, lineno, "json_decode_error"))
                continue

            row = normalize_row(raw_row, repo, path, file_locale)
            if row is None:
                skipped.append((repo, path, lineno, "missing_required_field"))
                continue

            out_rows[row["lang"]].append(row)
            skills.add(row["skill_id"])
            intents.add(row["expected_intent"])

    stats = {
        "lang_counts": {lang: len(rows) for lang, rows in sorted(out_rows.items())},
        "total": sum(len(rows) for rows in out_rows.values()),
        "n_skills": len(skills),
        "n_intents": len(intents),
        "skipped_count": len(skipped),
        "skipped_examples": skipped[:10],
    }
    return out_rows, stats


def validate_dataset(out_rows) -> list[str]:
    """Return a list of validation error strings; empty means valid."""
    errors = []
    for lang, rows in out_rows.items():
        if not LANG_RE.match(lang):
            errors.append(f"invalid lang tag: {lang!r}")
        for i, row in enumerate(rows):
            if not row.get("utterance"):
                errors.append(f"{lang}[{i}]: empty utterance")
            if not row.get("expected_intent"):
                errors.append(f"{lang}[{i}]: empty expected_intent")
            elif row["expected_intent"].endswith(".intent"):
                errors.append(f"{lang}[{i}]: expected_intent still has .intent suffix")
    return errors


def write_dataset(out_rows, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for lang, rows in sorted(out_rows.items()):
        lang_dir = os.path.join(out_dir, lang)
        os.makedirs(lang_dir, exist_ok=True)
        with open(os.path.join(lang_dir, "test.jsonl"), "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_readme(out_rows, pretty_name: str = "OVOS Golden Utterances Benchmark") -> str:
    langs = sorted(out_rows)
    primary_subtags = sorted({lang.split("-")[0].lower() for lang in langs})
    total = sum(len(rows) for rows in out_rows.values())

    front_matter = ["license: apache-2.0", "task_categories:", "- text-classification", "language:"]
    front_matter += [f"- {tag}" for tag in primary_subtags]
    front_matter.append(f"pretty_name: {pretty_name}")

    counts_table = ["| Language | Rows |", "|---|---|"]
    for lang in langs:
        counts_table.append(f"| {lang} | {len(out_rows[lang])} |")
    counts_table.append(f"| **Total** | **{total}** |")

    return f"""---
{chr(10).join(front_matter)}
---

# {pretty_name}

A held-out evaluation set of hand-curated golden utterances drawn from the
end-to-end test suites of the OpenVoiceOS skill fleet. Each row pairs an
utterance with the skill intent it is expected to trigger. The utterances
were written independently of the skills' own intent training templates
(vocab/intent files used to train the classifiers), so the set is suited to
measuring real intent-matching accuracy rather than re-testing the training
data itself.

## Schema

One JSONL file per language at `<lang>/test.jsonl`. Columns:

| Column | Type | Description |
|---|---|---|
| `utterance` | string | The natural-language phrase a user would say. |
| `expected_intent` | string | `<skill_id>:<intent_label>`, the intent the utterance must resolve to. |
| `lang` | string | BCP-47 locale of the utterance (e.g. `en-US`, `pt-PT`). |
| `skill_id` | string | The OVOS skill id that owns the intent. |
| `source_repo` | string | The `ovos-skill-*` GitHub repository the row was extracted from. |
| `source_file` | string | Path to the source golden-utterances file within that repo. |
| `intent_label_original` | string | The intent label exactly as it appears in the source suite, before normalization. |
| `intent_type` | string or null | The intent engine the utterance targets (e.g. `padatious`, `adapt`). |
| `intent_method` | string or null | The skill's handler method for the intent, when recorded. |
| `needs_manual` | bool | Whether the row is flagged as requiring manual review rather than automated scoring. |
| `machine_generated` | bool or null | Whether the utterance was machine-generated. `null` when the source row did not record this. |
| `required_vocab` | list or null | Vocabulary terms the utterance is expected to require, when recorded. |
| `expected_messages` | list or null | Bus message types the skill is expected to emit in response, when recorded. |

### Intent label normalization

Source suites label file-based (Padatious) intents with a trailing
`.intent` suffix, matching the `.intent` training file name, while runtime
dispatch strips that suffix before emitting the intent on the bus.
`expected_intent` always uses the stripped form, composed as
`<skill_id>:<intent_label without .intent>`. Adapt-era intents, which never
carried the suffix, compose unchanged. The original, unstripped label is
kept in `intent_label_original` for provenance.

Rows without a `lang` field in the source suite are the fleet's default,
`en-US`. Some suites ship as a single default-locale file per skill; others
ship one file per locale, named `golden_utterances_<locale>.jsonl` — in that
case every row in the file is stamped with that file's locale, unless the
row already carries its own `lang`, which wins.

### Flagged rows

Rows with `needs_manual: true` or `machine_generated: true` are included in
the dataset, not excluded, so that per-skill and per-language totals stay
complete. Consumers who want a strictly human-curated, auto-scorable subset
should filter both fields to `false`.

## Per-language row counts

{chr(10).join(counts_table)}

## License

Apache-2.0, matching the license of the source `ovos-skill-*` repositories.
"""


# --------------------------------------------------------------------------
# Publish
# --------------------------------------------------------------------------

def current_published_row_count(repo_id: str) -> int | None:
    """Best-effort total row count of the currently published dataset, or None."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return None
    try:
        api = HfApi()
        files = api.list_repo_files(repo_id, repo_type="dataset")
    except Exception:
        return None
    total = 0
    found_any = False
    for path in files:
        if not path.endswith("/test.jsonl"):
            continue
        try:
            from huggingface_hub import hf_hub_download

            local = hf_hub_download(repo_id, path, repo_type="dataset")
        except Exception:
            continue
        with open(local) as f:
            total += sum(1 for line in f if line.strip())
        found_any = True
    return total if found_any else None


def publish_dataset(out_dir: str, repo_id: str):
    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    who = api.whoami()
    print(f"Authenticated as: {who.get('name')}")

    create_repo(repo_id, repo_type="dataset", exist_ok=True)
    print(f"Repo ready: {repo_id}")

    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=out_dir,
        commit_message="Build OVOS golden-utterances evaluation dataset from skill-fleet end2end suites",
    )
    print("UPLOAD SUCCESS")
    print(f"https://huggingface.co/datasets/{repo_id}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="./golden-utterances-build", help="build output directory")
    parser.add_argument("--publish", action="store_true", help="upload the built dataset to the HF hub")
    parser.add_argument("--org", default=DEFAULT_ORG, help="GitHub org to enumerate skill repos from")
    parser.add_argument("--repo-filter", default=DEFAULT_REPO_FILTER, help="repo name prefix to include")
    parser.add_argument("--hf-repo-id", default=DEFAULT_HF_REPO_ID, help="HF dataset repo id to publish to")
    parser.add_argument("--force", action="store_true", help="publish even if row count dropped >20%%")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    session = _gh_session()
    print(f"Enumerating {args.org}/{args.repo_filter}* repos ...")
    repos = list_org_repos(args.org, args.repo_filter, session)
    print(f"Found {len(repos)} non-archived repos matching the filter")

    files, no_data = fetch_golden_files(args.org, repos, session)
    print(f"Fetched {len(files)} golden_utterances*.jsonl files "
          f"({len(no_data)} repos with no usable data)")

    out_rows, stats = build_dataset(files)

    print("\n=== Per-language row counts ===")
    for lang, count in sorted(stats["lang_counts"].items()):
        print(f"{lang}: {count}")
    print(f"\nTotal rows: {stats['total']}")
    print(f"Distinct skills: {stats['n_skills']}")
    print(f"Distinct expected_intent labels: {stats['n_intents']}")
    print(f"Skipped rows: {stats['skipped_count']}")
    for example in stats["skipped_examples"][:5]:
        print("  example skipped:", example)

    errors = validate_dataset(out_rows)
    if errors:
        print(f"\nVALIDATION FAILED ({len(errors)} errors):", file=sys.stderr)
        for err in errors[:20]:
            print(f"  {err}", file=sys.stderr)
        return 1

    write_dataset(out_rows, args.out)
    readme = render_readme(out_rows)
    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write(readme)
    print(f"\nBuild written to {args.out}")

    if not args.publish:
        return 0

    if not args.force:
        current_total = current_published_row_count(args.hf_repo_id)
        if current_total is not None:
            threshold = current_total * (1 - DROP_GUARD_FRACTION)
            if stats["total"] < threshold:
                print(
                    f"\nREFUSING TO PUBLISH: new total {stats['total']} rows is more than "
                    f"{DROP_GUARD_FRACTION:.0%} below the currently published "
                    f"{current_total} rows. Pass --force to override.",
                    file=sys.stderr,
                )
                return 1

    publish_dataset(args.out, args.hf_repo_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
