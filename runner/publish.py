"""
Publish completed JSONL output files to HuggingFace datasets.

Appends new files (shard naming: ``stt_<lang>_<plugin>_<n>.jsonl``)
to the target HF dataset repo.  Idempotent: already-uploaded shards
(by filename stem) are skipped.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Maximum rows per uploaded shard
_SHARD_SIZE = 100_000


def _next_shard_index(api, repo_id: str, stem: str) -> int:
    """Return the next available shard index for this stem."""
    try:
        siblings = api.list_repo_files(repo_id, repo_type="dataset")
    except Exception:
        return 0
    pattern = re.compile(rf"^{re.escape(stem)}_(\d+)\.jsonl$")
    indices = [
        int(m.group(1))
        for f in siblings
        if (m := pattern.match(Path(f).name))
    ]
    return max(indices, default=-1) + 1


def publish_output(
    output_file: Path,
    hf_repo: str,
    token: Optional[str] = None,
    commit_message: Optional[str] = None,
) -> list[str]:
    """
    Upload *output_file* to *hf_repo* as one or more shards.

    Returns list of uploaded filenames.
    """
    from huggingface_hub import HfApi

    if not output_file.exists() or output_file.stat().st_size == 0:
        logger.warning("publish skipped: %s is empty or missing", output_file)
        return []

    api = HfApi(token=token)

    # Derive a stem from the filename: everything before the last _N.jsonl
    # e.g. stt_pt-PT_ovos_stt_plugin_fasterwhisper_small -> that
    stem = output_file.stem  # already without extension
    # Strip any trailing numeric shard index left from a previous partial run
    stem = re.sub(r"_\d+$", "", stem)

    # Split into shards if needed
    lines: list[str] = []
    with output_file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                lines.append(line)

    if not lines:
        logger.warning("publish skipped: %s has no valid lines", output_file)
        return []

    start_idx = _next_shard_index(api, hf_repo, stem)
    uploaded: list[str] = []

    for shard_n, offset in enumerate(range(0, len(lines), _SHARD_SIZE)):
        chunk = lines[offset: offset + _SHARD_SIZE]
        remote_name = f"{stem}_{start_idx + shard_n}.jsonl"
        content = "\n".join(chunk) + "\n"

        api.upload_file(
            path_or_fileobj=content.encode("utf-8"),
            path_in_repo=remote_name,
            repo_id=hf_repo,
            repo_type="dataset",
            commit_message=commit_message or f"runner: add {remote_name}",
        )
        logger.info("uploaded %s to %s", remote_name, hf_repo)
        uploaded.append(remote_name)

    return uploaded
