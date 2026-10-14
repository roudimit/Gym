# Tau-Voice Simple agent

The candidate loop for the `tau_voice_simple` resources server; see
`resources_servers/tau_voice_simple/README.md` for the benchmark.

For each episode the agent seeds a session on the resources server, then:

1. Appends the latest observation (a user WAV clip, followed by the fixed
   listening instruction, or the tool results) to the candidate's history.
2. Calls the policy model's `/v1/chat/completions` with the full history, the
   domain tools (`tool_choice: auto`), and the row's sampling settings. Each
   attempt uses seed `row seed + attempt`.
3. Accepts exactly one tool call or one spoken reply. Other outputs are
   retried (`protocol_retries`, default 2) and then replaced by the
   reference's fallback reply.
4. Submits the turn to `/agent_turn` and repeats until Tau ends the episode,
   then calls `/verify`.

Reasoning is removed from replies and never fed back to the candidate. The
rollout transcript shows each user clip as a `[user audio turn NNNN]`
placeholder. The clips themselves are in the resources server's audio
archive.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
