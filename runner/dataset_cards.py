"""The funding block every published dataset card carries, and the check that
keeps it there.

The arena's prediction datasets are public evidence for grant-funded work, so
each card names the funder. Naming it is an obligation on the artifact, not a
formatting preference, and it has been lost once already: a pass that rewrote
every card to set its licence replaced the whole card and dropped the block with
it, months after both publishers had been writing it correctly.

Defining the block in one place and asserting it on the way out is what makes
that failure loud. A card that reaches :func:`check_funding_block` without the
block does not reach the Hub.
"""
from __future__ import annotations

FUNDING_BLOCK = """Funded by the [NGI0 Commons Fund](https://nlnet.nl/project/OpenVoiceOS) /
[NLnet](https://nlnet.nl) under grant agreement No
[101135429](https://cordis.europa.eu/project/id/101135429), through the
European Commission's [Next Generation Internet](https://ngi.eu) programme."""


class MissingFundingBlock(RuntimeError):
    """A card was about to be published without naming the funder."""


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


def _front_matter(pretty_name: str, kind: str) -> str:
    """Card front matter, tagged with the path that produced it.

    The ``card-<kind>`` tag is how a reader tells a repo whose card came from
    the sample-set publisher from one written by a prediction publisher,
    without diffing the bodies.
    """
    return (
        "---\n"
        "license: apache-2.0\n"
        "tags:\n"
        "  - openvoiceos\n"
        "  - benchmark\n"
        f"  - card-{kind}\n"
        f"pretty_name: {pretty_name}\n"
        "---\n"
    )


def sample_set_card(dataset_id: str, lang: str, hf_id: str) -> str:
    """Card for a repo holding a published sample-set manifest.

    A manifest repo carries no predictions. Saying so is the point: the same
    repo later gains prediction shards, and a reader who arrives early should
    not be left to guess whether the benchmark failed or has not run.
    """
    return check_funding_block(f"""{_front_matter(
        f"OVOS sample sets — {dataset_id}", "sample-set-manifest")}
# OVOS sample sets — `{dataset_id}`

The rows of [`{hf_id}`](https://huggingface.co/datasets/{hf_id}) that every
fighter on this benchmark is scored over, published as
`sample_sets/<lang>.json` and pinned to one corpus revision.

Fixing the sample set is what makes two fighters comparable: without it, a
fighter that ran on a different draw of the corpus would be ranked beside one
that did not, and the difference in scores would be partly the draw.

This repository holds manifests rather than predictions. Prediction rows for
`{dataset_id}` live in the benchmark repository for their own modality.

{FUNDING_BLOCK}
""", f"sample-set card for {dataset_id}/{lang}")
