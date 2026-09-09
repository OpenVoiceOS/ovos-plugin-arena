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
