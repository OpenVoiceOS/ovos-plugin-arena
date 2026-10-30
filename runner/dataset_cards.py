"""What every published prediction dataset card must say, and the checks that
keep it there.

The arena's prediction datasets are public evidence for grant-funded work, so
each card names the funder. Naming it is an obligation on the artifact, not a
formatting preference, and it has been lost once already: a pass that rewrote
every card to set its licence replaced the whole card and dropped the block with
it, months after both publishers had been writing it correctly.

The same pass set every card to ``apache-2.0``. That is wrong for the rows: a
prediction row repeats the test set's reference text (an intent utterance, an
STT reference transcript, a TTS prompt), so the rows carry the licence of the
test set, not the licence of the arena's code. A non-commercial (NC) or
share-alike (SA) licence permits that, because the rows are published free of
charge under the same licence. A no-derivatives (ND) licence does not, so
:func:`check_redistributable` refuses the upload when a row carries text from
such a set.

Defining the blocks in one place and asserting them on the way out is what
makes these failures loud. A card that reaches :func:`check_funding_block`
without the funding block does not reach the Hub.

Run ``python -m runner.dataset_cards`` to see the licence each existing
prediction repo card should carry; ``--apply`` rewrites the cards on the Hub.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

FUNDING_BLOCK = """Funded by the [NGI0 Commons Fund](https://nlnet.nl/project/OpenVoiceOS) /
[NLnet](https://nlnet.nl) under grant agreement No
[101135429](https://cordis.europa.eu/project/id/101135429), through the
European Commission's [Next Generation Internet](https://ngi.eu) programme."""

#: Hugging Face licence ids a dataset card's ``license:`` field accepts, for
#: the data licences the registry uses or is likely to use.
HF_LICENCES = frozenset({
    "apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "isc", "unlicense",
    "cc0-1.0", "pddl", "odc-by", "odbl",
    "cc-by-2.0", "cc-by-2.5", "cc-by-3.0", "cc-by-4.0",
    "cc-by-sa-3.0", "cc-by-sa-4.0",
    "cc-by-nd-4.0",
    "cc-by-nc-2.0", "cc-by-nc-3.0", "cc-by-nc-4.0",
    "cc-by-nc-sa-2.0", "cc-by-nc-sa-3.0", "cc-by-nc-sa-4.0",
    "cc-by-nc-nd-3.0", "cc-by-nc-nd-4.0",
    "cdla-permissive-1.0", "cdla-permissive-2.0", "cdla-sharing-1.0",
    "gpl-3.0", "lgpl-3.0", "agpl-3.0", "mpl-2.0", "other",
})

#: Prediction row fields that repeat text from the test set.
TEXT_FIELDS = ("utterance", "reference_text", "input_text")

#: Leagues whose rows carry one of :data:`TEXT_FIELDS`.
TEXT_LEAGUES = frozenset({"intent", "intent_template", "intent_keyword", "stt", "tts"})

LICENCE_HEADING = "## Licence"


class MissingFundingBlock(RuntimeError):
    """A card was about to be published without naming the funder."""


class RestrictedLicence(RuntimeError):
    """Rows were about to republish text from a no-derivatives licensed set."""


class CardFormatError(ValueError):
    """An existing card has front matter the rewrite cannot edit safely."""


def check_funding_block(card: str, repo: str) -> str:
    """Return *card* unchanged, or refuse to let it out.

    Called by every path that writes a README to the Hub, so a card that loses
    the block fails where it is written rather than where a reviewer finds it.
    """
    if FUNDING_BLOCK not in card:
        raise MissingFundingBlock(
            f"{repo}: card does not name the NGI0 Commons Fund, NLnet or grant "
            f"agreement No 101135429. Build it from runner.media_bench."
            f"dataset_card or runner.intent_bench._dataset_card rather than by "
            f"hand."
        )
    return card


def hf_licence(raw: str | None) -> str:
    """The Hugging Face licence id for a registry ``license`` value.

    A value with no Hugging Face id, or no value at all, becomes ``other``;
    :func:`licence_section` then states the original value in the card text.
    """
    if not raw:
        return "other"
    normalised = re.sub(r"[\s_]+", "-", raw.strip().lower())
    return normalised if normalised in HF_LICENCES else "other"


def _has_term(raw: str | None, short: str, full: str) -> bool:
    """True when *raw* holds the short code as a word (``nd``) or the full
    name in any spelling (``NoDerivatives``, ``No-Derivatives``,
    ``no derivs``). Matching the joined letters errs towards a match."""
    if not raw:
        return False
    lowered = raw.lower()
    return (short in re.split(r"[^a-z]+", lowered)
            or full in re.sub(r"[^a-z]", "", lowered))


def forbids_derivatives(raw: str | None) -> bool:
    """True when a licence forbids derivatives (ND)."""
    return _has_term(raw, "nd", "noderiv")


def is_noncommercial(raw: str | None) -> bool:
    """True when a licence forbids commercial use (NC)."""
    return _has_term(raw, "nc", "noncommercial")


def is_share_alike(raw: str | None) -> bool:
    """True when a licence requires adaptations to keep it (SA)."""
    return _has_term(raw, "sa", "sharealike")


def licence_section(eval_def: Any) -> str:
    """The card section that names the source set, its licence and its credit."""
    hf_id = eval_def.source.hf_id
    source = f"[{eval_def.display_name or hf_id}](https://huggingface.co/datasets/{hf_id})"
    raw = eval_def.license
    if not raw:
        terms = ("The registry records no licence for this set. Read the "
                 "source card for its terms before you reuse the rows.")
    elif hf_licence(raw) == "other":
        terms = (f"The card gives `other` as the Hugging Face licence id. The "
                 f"source licence is: `{raw}`. Read the source card for its "
                 f"terms before you reuse the rows.")
    else:
        terms = f"The rows carry the licence of that set, `{hf_licence(raw)}`."
    if is_noncommercial(raw):
        terms += (" The licence forbids commercial use: reuse the rows for "
                  "non-commercial purposes only.")
    if is_share_alike(raw):
        terms += " Share any adaptation of the rows under the same licence."
    return f"""{LICENCE_HEADING}

The rows are predictions over the test set {source} and repeat parts of it,
such as its reference text or labels. {terms} The Apache-2.0 licence of the
arena's code does not cover these rows.

Credit the authors of `{hf_id}` when you reuse the rows, and cite the set as
its own card asks."""


def rows_with_text(paths: Iterable[Path]) -> Iterator[tuple[Path, str]]:
    """Yield ``(file, field)`` for each JSONL row in *paths* that carries
    text from the test set."""
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                field = next((f for f in TEXT_FIELDS if row.get(f)), None)
                if field:
                    yield path, field


def check_redistributable(eval_def: Any, rows: Path | Iterable[Path], repo: str) -> None:
    """Refuse to publish rows that repeat text from a no-derivatives set.

    *rows* is a directory searched for JSONL files, or the files themselves.
    Non-commercial and share-alike sets publish, under their own licence. Rows
    that carry no text from the set, such as voice-activity labels and links
    to the source clips, publish whatever the licence.
    """
    if not forbids_derivatives(eval_def.license):
        return
    paths = sorted(rows.rglob("*.jsonl")) if isinstance(rows, Path) else list(rows)
    found = next(rows_with_text(paths), None)
    if found:
        path, field = found
        raise RestrictedLicence(
            f"{repo}: {eval_def.dataset_id} is licensed {eval_def.license}, "
            f"which forbids derivatives, and {path.name} republishes its "
            f"text in '{field}'. Publishing refused."
        )


def check_repo_redistributable(repo: str, rows: Path | Iterable[Path]) -> None:
    """:func:`check_redistributable` for a caller that knows only the repo.

    A repo that no registry dataset names has no known licence, so its rows
    are refused when they carry text.
    """
    from registry.loaders import HF_OWNER, load_all_datasets

    eval_def = repo_index(load_all_datasets(), HF_OWNER).get(repo)
    if eval_def is None:
        eval_def = SimpleNamespace(dataset_id=repo, license="unknown, treated as ND")
    check_redistributable(eval_def, rows, repo)


#: A top-level ``license:`` key with its value, scalar or block list.
_LICENSE_KEY = re.compile(r"^license:[^\n]*(?:\n[ \t]*-[^\n]*)*", flags=re.M)


def _set_front_matter_licence(card: str, licence: str, repo: str) -> str:
    if not card.startswith("---\n"):
        return f"---\nlicense: {licence}\n---\n\n{card}"
    end = card.find("\n---", 3)
    if end == -1:
        raise CardFormatError(f"{repo}: card front matter has no closing '---'")
    front = card[3:end]
    if _LICENSE_KEY.search(front):
        front = _LICENSE_KEY.sub(f"license: {licence}", front, count=1)
    else:
        front = f"\nlicense: {licence}{front}"
    return f"---{front}{card[end:]}"


def relicensed_card(card: str, eval_def: Any, repo: str) -> str:
    """Return *card* with the source set's licence in its front matter and a
    licence section, keeping every other part of the card.

    An existing licence section is replaced. A card that lost the funding
    block gets it back, below the licence section. CRLF line ends become LF.
    Raises :class:`CardFormatError` when the front matter is not closed.
    """
    card = _set_front_matter_licence(card.replace("\r\n", "\n"),
                                     hf_licence(eval_def.license), repo)
    start = card.find(f"\n{LICENCE_HEADING}\n")
    if start != -1:
        after = start + len(LICENCE_HEADING) + 2
        ends = [i for i in (card.find("\n## ", after), card.find(FUNDING_BLOCK, after))
                if i != -1]
        end = min(ends) if ends else len(card)
        card = card[:start].rstrip("\n") + "\n\n" + card[end:].lstrip("\n")
    section = licence_section(eval_def)
    if FUNDING_BLOCK in card:
        head, tail = card.split(FUNDING_BLOCK, 1)
        card = f"{head.rstrip()}\n\n{section}\n\n{FUNDING_BLOCK}{tail}"
    else:
        card = f"{card.rstrip()}\n\n{section}\n\n{FUNDING_BLOCK}\n"
    return check_funding_block(card, repo)


def card_licence(card: str) -> str | None:
    """The ``license:`` value of a card's front matter, if it has one."""
    if not card.startswith("---\n"):
        return None
    front = card[3:card.find("\n---", 3)]
    match = re.search(r"^license:[ \t]*(\S*)", front, flags=re.M)
    return (match.group(1) or "list") if match else None


def repo_index(datasets: Iterable[Any], owner: str) -> dict[str, Any]:
    """Map every prediction repo name the publishers write to its source set.

    An intent eval set publishes three repos, one per supervision league. An
    eval set wins over a training set that names the same repo.
    """
    from registry.loaders import results_repo_for

    index: dict[str, Any] = {}
    for d in sorted(datasets, key=lambda d: d.role != "eval"):
        modality = d.modality.value
        leagues = ([modality, "intent_template", "intent_keyword"]
                   if modality == "intent" else [modality])
        names = [results_repo_for(m, d.dataset_id, owner) for m in leagues]
        if d.predictions_hf:
            names.append(d.predictions_hf)
        for name in names:
            index.setdefault(name, d)
    return index


def _published_rows_note(api: Any, repo: str) -> str:
    """Read one row file of *repo* and say whether it republishes text."""
    from huggingface_hub import hf_hub_download

    files = sorted(f for f in api.list_repo_files(repo, repo_type="dataset")
                   if f.endswith(".jsonl"))
    if not files:
        return "no rows published"
    local = Path(hf_hub_download(repo, files[0], repo_type="dataset"))
    found = next(rows_with_text([local]), None)
    return (f"{files[0]} republishes text in '{found[1]}'; withdraw the rows"
            if found else f"{files[0]} carries no text")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Set the licence of every existing prediction repo card "
                    "to the licence of its source test set. Prints the "
                    "planned change per repo; writes nothing without --apply.")
    parser.add_argument("--apply", action="store_true",
                        help="Upload the rewritten cards (default: dry run)")
    parser.add_argument("--hf-owner", default="OpenVoiceOS")
    parser.add_argument("--repo", action="append", default=[],
                        help="Only this repo (repeatable)")
    args = parser.parse_args(argv)

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    from registry.loaders import load_all_datasets

    api = HfApi()
    index = repo_index(load_all_datasets(), args.hf_owner)
    pattern = re.compile(rf"{re.escape(args.hf_owner)}/ovos-[a-z-]+-bench-.+")
    listed = sorted(d.id for d in api.list_datasets(author=args.hf_owner)
                    if pattern.fullmatch(d.id))
    repos = [r for r in listed if not args.repo or r in args.repo]
    print(f"mode: {'apply' if args.apply else 'dry-run'}; "
          f"{len(repos)} prediction repos listed under {args.hf_owner}")

    planned: Counter[str] = Counter()
    unmatched: list[str] = []
    failed: list[str] = []
    for repo in repos:
        eval_def = index.get(repo)
        if eval_def is None:
            unmatched.append(repo)
            continue
        try:
            try:
                old = Path(hf_hub_download(repo, "README.md", repo_type="dataset")
                           ).read_text(encoding="utf-8")
            except EntryNotFoundError:
                old = ""
            new = relicensed_card(old, eval_def, repo)
            note = (f" NO-DERIVATIVES: {_published_rows_note(api, repo)}"
                    if forbids_derivatives(eval_def.license) else "")
            if args.apply and new != old:
                api.upload_file(
                    path_or_fileobj=new.encode(), path_in_repo="README.md",
                    repo_id=repo, repo_type="dataset",
                    commit_message="docs: set the card licence to the source "
                                   "test set's licence")
        except Exception as exc:
            failed.append(repo)
            print(f"{repo}: ERROR, card left as it is: {exc}")
            continue
        licence = hf_licence(eval_def.license)
        planned[licence] += 1
        print(f"{repo}: {card_licence(old) or 'none'} -> {licence} "
              f"[source {eval_def.dataset_id}, registry {eval_def.license!r}] "
              f"{'unchanged' if new == old else 'rewrite'}{note}")

    matched = sum(planned.values())
    print(f"\n{len(repos)} in, {matched} matched and done, {len(failed)} failed, "
          f"{len(unmatched)} unmatched")
    for licence, count in sorted(planned.items()):
        print(f"  {licence}: {count}")
    if unmatched:
        print("unmatched (no registry dataset; left untouched):")
        for repo in unmatched:
            print(f"  {repo}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
