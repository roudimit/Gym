# BFCL v3 resource server

Scores single-turn function calling from the Berkeley Function Calling
Leaderboard v3 categories `simple`, `multiple`, `parallel`,
`parallel_multiple`, and `irrelevance`, with Gorilla's AST checker and no
judge. The same server scores text rows and audio rows (the question spoken as
a WAV); only the rows differ.

Scoring matches VLMEvalKitMcore (`vlmeval/dataset/avlm/bfclv3.py`, commit
`df7a6995`, prompt style `template`):

- **Prediction**: the text after the last `</think>` of
  `<think>\n{reasoning}\n</think>\n{content}`; with reasoning enabled, a reply
  without reasoning becomes an empty prediction (no calls).
- **Parsing**: `<think>` blocks are dropped, then every
  `<function=NAME><parameter=KEY>value</parameter></function>` is read, inside
  `<tool_call>` blocks when present. Values are `ast.literal_eval`ed when they
  parse and kept as strings otherwise. Several `<tool_call>` blocks form one
  call list.
- **Checking**: tools are normalized (`"type": "dict"` becomes `"object"`), and
  each gold argument becomes a list of accepted values (optional parameters also
  accept `""`, meaning omitted). Gorilla's `ast_checker` then decides the row;
  both parallel categories use its order-free `parallel` checker. An
  `irrelevance` row passes when the candidate makes no call. The checker is
  vendored verbatim in `gorilla/ast_checker.py`; see `gorilla/VENDORING.md`.
- **Reward**: 1.0 when the checker accepts the row, else 0.0.
- **Metrics**: `bfclv3/ast_accuracy` is the micro average over the
  non-irrelevance rows (the reference headline);
  `bfclv3/irrelevance_accuracy` and `bfclv3/irrelevance_false_positive_rate`
  cover the irrelevance rows. Also reported: name accuracy, per-category
  accuracy, parse counts, and coverage.

Rows carry the tools only in the prompt: the system message is the chat
template's own rendering of the tools (the `# Tools` block with the
`<tool_call>` format), so the policy server needs no tool-call parser and no
`tools` request parameter. `verifier_metadata` holds `category`, `tools`,
`gold_toolcall`, and, for audio rows, the transcript `user_text`, which is
never shown to the candidate.

Audio rows reference their WAV in `responses_create_params.metadata.audio_refs`
(one group with one `{"archive": ..., "member": ...}`), resolved against the
`vllm_model` adapter's `audio_root`; the adapter must use Chat Completions
(`is_responses_native: false`, `use_completions_api: false`).
`data/example.jsonl` shows one audio row per category; its audio archive is not
part of this repository.
