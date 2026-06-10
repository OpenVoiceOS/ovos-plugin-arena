"""
OVOS Plugin Arena — STT Prediction Runner

Offline batch runner that generates STT predictions for one or more
(plugin, dataset, lang) jobs and appends rows to the corresponding
HuggingFace benchmark dataset (``OpenVoiceOS/ovos-stt-bench-<lang>``).

Row schema matches the real ``ovos-stt-bench-*`` column layout accepted
by the arena ingestion layer (§3.2 compatibility note in SPECIFICATION.md):

    dataset_entry_id   – stable filename within source corpus
    plugin_name        – OPM entry-point name
    model_id           – composite ``plugin/model/hash`` identifier
    prediction_transcript – STT output text
    transcript         – ground truth text
    prediction_confidence – model confidence score (0.0–1.0)
    prediction_type    – always "STT"

See ``runner/queue.yaml`` for job definitions and ``docs/runner.md``
for deployment and monitoring instructions.
"""
