# AudioMC resource server

AudioMC evaluates multi-turn spoken conversations. Each task is a conversation
of 3–8 user turns, each with its own audio clip, followed by the candidate's
answer to the last turn. An LLM judge checks the answer against each of the
task's rubric items independently.

Scoring matches VLMEvalKitMcore (`vlmeval/dataset/avlm/audiomc.py`, commit
`df7a6995`):

- **Prediction**: the text after the last `</think>` of
  `<think>\n{reasoning}\n</think>\n{content}`; with reasoning enabled, a reply
  without reasoning becomes an empty prediction. An empty prediction fails
  every rubric item without judge calls.
- **Judge**: one call per rubric item with the versioned prompt
  `prompts/legacy_rubric_v1.yaml` (template in the system role,
  `"Return the JSON judgment now."` as user). The reply's JSON
  `criteria_met` decides the item; replies are retried up to `judge_attempts`
  times and an item that still fails is counted as not met. Reference profile:
  `azure/openai/o4-mini`, temperature 0.0, 4096 max tokens, 3 attempts.
- **Reward**: 1.0 when every rubric item is met (task pass), else 0.0.
- **Metrics**: `audiomc/apr` (fraction of tasks passing) and `audiomc/ars`
  (fraction of all rubric items met), overall and per axis
  (`INFERENCE_MEMORY`, `INSTRUCTION_RETENTION`, `SELF_COHERENCE`,
  `VOICE_EDITING`), plus coverage and judge-error counts.

Rows carry one audio group per user message in
`responses_create_params.metadata.audio_refs`, a JSON string of
`[[{"archive": ..., "member": ...}], ...]` resolved against the `vllm_model`
adapter's `audio_root`. The adapter must use Chat Completions
(`is_responses_native: false`, `use_completions_api: false`); set
`audio_content_order: before_text` so the final instruction follows the last
clip, and allow at least 8 audio inputs per prompt on the server.

`data/example.jsonl` shows three prepared rows; their audio archives are not
part of this repository.
