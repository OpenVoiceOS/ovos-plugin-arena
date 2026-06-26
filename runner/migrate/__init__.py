"""One-shot migrators that re-key legacy benchmark predictions into the §3.2
prediction contract, to bootstrap arena boards and ELO seeds.

Migrated rows are clearly tagged (``migrated: true``, ``runner_version`` names
the source) so they are distinguishable from native arena runs and can be
replaced by a reproducible sweep later (§P6 keeps all prediction data).
"""
