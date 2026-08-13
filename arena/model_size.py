"""Model download size lookup (M2 — performance boards, §2 model size).

A fighter's ``model_hf_repo`` (``registry.schemas.CompetitorDef``, optional)
points at the HuggingFace model repo its weights ship from. The download
size — sum of every file in that repo — is fetched from HF metadata ONCE
per fighter and cached for the life of the build process (``export-bestiary``
/ ``assemble``), never per prediction row: it is a property of the model
artifact, not of any single benchmark run.

Fighters with no ``model_hf_repo`` (rule-based engines, sklearn pipelines
shipped as plugin code, etc.) get ``None`` — never 0, which would misleadingly
read as "no download at all" instead of "not tracked".
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

#: Shape check for "this string looks like a HF repo id" (`owner/name`,
#: alphanumeric + '-', '_', '.' only — the same charset HfApi itself
#: validates against). Used as a fallback source for a fighter's model
#: repo when ``model_hf_repo`` isn't set (see ``likely_hf_repo_id``) — many
#: fighters carry their HF repo id in the pre-existing ``model`` field
#: (e.g. ``"OpenVoiceOS/whisper-tiny-onnx"``) but that field is free text
#: for other modalities/engines (voice ids, coqui model paths, gTTS
#: descriptions with spaces/parens, …), so only a strict owner/name shape
#: qualifies — never guess on anything looser.
_HF_REPO_ID_RE = re.compile(r"^[\w.\-]+/[\w.\-]+$")


def likely_hf_repo_id(value: str | None) -> bool:
    """Whether *value* has the ``owner/name`` shape of a HF repo id.

    A cheap, local, no-network shape check — it does NOT confirm the repo
    exists (``model_repo_size_mb`` already tolerates a 404/lookup failure
    by returning None). Deliberately strict (no spaces, no parens, exactly
    one ``/``) so free-text ``model`` values like ``"gTTS en (tld=us)"`` or
    ``"YatharthS/LuxTTS (zipvoice)"`` are rejected rather than sent to HF
    as a wasted/garbage lookup.
    """
    if not value or not isinstance(value, str):
        return False
    return bool(_HF_REPO_ID_RE.match(value))

#: Build-process-lifetime cache: repo_id -> size in MB (or None on failure).
#: Populated by ``model_repo_size_mb``; a fresh process gets a fresh cache
#: (this deliberately does NOT persist to disk — HF repo contents can
#: change, and a build should reflect current metadata, not a stale value
#: from a previous invocation).
_SIZE_CACHE_MB: dict[str, float | None] = {}


def _fetch_model_repo_size_mb(repo_id: str, revision: str = "main") -> float | None:
    """Sum of every file's size in a HF *model* repo, in MB. None on any
    lookup failure (repo missing, network error, no size metadata) — a
    build must never crash because one fighter's model repo is unreachable.
    """
    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    except Exception:
        log.warning("model_size: could not fetch metadata for %r", repo_id, exc_info=True)
        return None

    siblings = info.siblings or []
    total_bytes = 0
    known = False
    for sib in siblings:
        size = getattr(sib, "size", None)
        if size is None:
            lfs = getattr(sib, "lfs", None)
            size = getattr(lfs, "size", None) if lfs is not None else None
        if size is not None:
            total_bytes += size
            known = True
    if not known:
        return None
    return round(total_bytes / (1024 * 1024), 2)


def model_repo_size_mb(
    repo_id: str | None, revision: str = "main", *, cache: dict[str, float | None] | None = None
) -> float | None:
    """Cached MB size of a fighter's model repo, or None when *repo_id* is
    falsy or the lookup fails. *cache* defaults to the module-level
    build-lifetime cache; pass an explicit dict in tests to avoid leaking
    state across test cases."""
    if not repo_id:
        return None
    store = _SIZE_CACHE_MB if cache is None else cache
    key = f"{repo_id}@{revision}"
    if key in store:
        return store[key]
    size = _fetch_model_repo_size_mb(repo_id, revision)
    store[key] = size
    return size


def reset_cache() -> None:
    """Clear the module-level size cache (tests only)."""
    _SIZE_CACHE_MB.clear()
