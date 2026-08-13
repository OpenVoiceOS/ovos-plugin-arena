#!/usr/bin/env python3
"""G2P benchmark (§A6, §P4): TigreGotico portuguese_phonetic_lexicon, pt-PT.

Thin wrapper around :mod:`runner.g2p_bench` — one dedicated script per
benchmark, per the P4 convention every other league follows.
"""
from __future__ import annotations

from runner.g2p_bench import build_arg_parser, run

DATASET_ID = "portuguese-phonetic-lexicon-pt-PT"


def main(argv=None) -> int:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = build_arg_parser()
    parser.set_defaults(dataset=DATASET_ID)
    args = parser.parse_args(argv)
    return run(args, DATASET_ID)


if __name__ == "__main__":
    raise SystemExit(main())
