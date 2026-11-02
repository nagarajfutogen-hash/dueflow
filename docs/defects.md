# Defects found and fixed

Real defects, how they were found, and the test that keeps them fixed. All data is
SYNTHETIC; no production calls exist.

## D-001 — LLM invented a date of birth for identity verification

**Found:** 2026-10-24, Day 2 live smoke run (Groq `openai/gpt-oss-20b`, 5 scripted
conversations, one run each).

**What happened.** The `verify_identity` tool took `full_name` and `date_of_birth` as LLM
arguments. When the debtor said only *"Yes, this is Live Promise."*, the LLM called
`verify_identity` with a well-formed `YYYY-MM-DD` date the debtor never said.

- It could not guess the real date (the DOB is never in the prompt), so the attempt failed.
- But each invented date **burned one of the two allowed attempts**. In the `wrong_person`
  smoke test the call ended after only one real attempt, so the test **passed for the wrong
  reason**.
- The general risk: an LLM filling a security-relevant tool argument with invented values,
  or a debtor's prompt injection steering what the LLM passes.

**How it was found.** All 5 conversations reached the expected outcome, so outcome checks
alone missed it. Reading the transcripts showed a retry message right after "Yes, speaking";
the event trail showed an `identity_mismatch` with no date in the debtor's words; the server
log (redacted check) confirmed the date argument was not something the debtor said.

**Fix (code, not prompt).**
- `verify_identity` takes **no arguments**. The LLM only signals "details were given".
- Code checks the **debtor's own words** since the last attempt
  (`app/policy/identity.py: find_dates, name_in_text`; `check_identity(debtor_text=...)`).
- No date stated at all → `no_details`: **not counted** as an attempt; the LLM is told to ask.
- Any stated date that is wrong → counted attempt (`identity_mismatch`).
- Follow-up: the identify node ends politely after 4 debtor turns without a counted attempt
  (`no_verification`), so "never stating a date" cannot stall a call.

**Tests.**
- `tests/test_chat_offline.py::test_llm_supplied_identity_values_are_ignored`
- `tests/test_identity.py::test_no_date_stated_is_not_counted`
- `tests/policy/test_identity_rules.py::test_no_date_stated_is_not_an_attempt`
- `tests/test_chat_offline.py::test_identify_turn_cap_ends_with_no_verification`
- Live re-run of `wrong_person`: agent asked for details; call ended after **two real**
  attempts (2,112 Groq tokens).

**Lesson.** Outcome-only checks can pass for the wrong reason; evals must also check the
event trail (here: every counted identity attempt must contain a date the debtor said).

## Found in the same run

| ID | Defect | Fix | Test |
|---|---|---|---|
| D-002 | Interceptor blocked a valid rule figure ("pay in full within **14** days") | Rule limits mentioned in the prompt are allowed figures | `test_output_guard.py::test_rule_limits_in_the_prompt_are_allowed` |
| D-003 | Pipecat DEBUG logs printed full LLM context (names, DOBs) | Pipecat log level WARNING; old container logs cleared | manual: 15 PII log lines before, 0 after recreate |
| D-004 | WebSocket teardown could be cancelled mid-shutdown (flaky test, ~1 in 8) | Shielded teardown; hang-up recorded as `disconnected` | WS tests 25/25, full suite repeated |
| D-005 | D-003 again, via a new entry point: the eval CLI never imported `app.main`, so Pipecat ran at DEBUG and printed transcripts during the Day 3 dry run | Log level moved to `app/logging_setup.py`, called by every entry point | `test_evals_offline.py::test_every_entry_point_configures_pipecat_logging` |

## D-006 — v2 code-run identity check left the LLM a context it misread as a plan

**Found:** 2026-10-25, Day 4 Phase 4, v2 eval run #10 (base, `sim-v2`, Groq
`openai/gpt-oss-20b`, k=1), while the run was still going.

**What happened.** v2 fix `code_identity_check` runs `verify_identity` in code as soon as the
debtor's own words contain a date. On success it switched straight to the negotiate node. The
LLM context then ended with the debtor's name + date of birth, followed by the negotiate task.
The LLM's first move was a `propose_terms` call nobody asked for ("0 installments, ¥0 down"),
so the agent opened with "that option isn't available" instead of stating the balance.

- In v1 (tool path) the context ends with the LLM's own `verify_identity` call and its result.
- `base/disputes_debt` passed in v1 and failed in v2: after the bad opening the agent never
  offered the dispute handoff, and the call ended as `stop_contact`.

**How it was found.** Watching run #10 per scenario: an outcome that passed in v1 failed in
v2. The transcript showed the unprompted refusal right after "identity is confirmed". The first
agent line after verification, compared across runs: v1 run #7 stated the balance in 6 of 6
conversations; v2 run #10 called `propose_terms` first in 3 of 4.

**Cause (likely, from transcripts and events):** the LLM read the debtor's last message
(name + date of birth) as an answer to the negotiate task. This is not confirmed by a replay.

**Handling.** Run #10 (base, complete) and run #11 (adversarial, 1 scenario) were stopped
and kept on record with `status = aborted`, `status_reason = "D-006: v2 identity fix defect"`
(migration `a1b2c3d4e5f6`). `evals.compare` and `evals.gate` refuse aborted runs. v2 had not
been reported yet, so it was fixed in place and re-run on all 44 scenarios.

**Fix (code).** The code path now writes into the LLM context exactly what the tool path
would: an assistant `verify_identity` tool call plus its tool result, then the node switch.
Retry / ended results don't run the LLM (code already said the line).

**Tests.**
- `tests/test_agent_versions.py::test_v2_code_check_leaves_the_same_context_as_the_tool_path`
- `tests/test_agent_versions.py::test_v2_wrong_details_are_a_counted_attempt`
- `tests/test_compare.py::test_refuses_aborted_runs`,
  `tests/test_gate.py::test_aborted_runs_are_never_gated_or_exported`

**Lesson.** A code path that replaces an LLM decision must also leave the LLM the same context
the decision would have left. Offline tests checked the audit trail and the node switch, but not
the shape of the LLM's context.

**Also seen in run #10 (not D-006):** `base/angry` ended `disconnected`: the simulator output
`"Yeah, it's me.[HANGUP]"` right after confirming identity. Its goal is to complain, then agree.
Classified as a **simulator fault** under the fault criteria in `docs/eval-v1-report.md`. Not
fixed mid-comparison: v1 and v2 must share the simulator version. v2's wrong-number guidance did
hold in that conversation (the agent asked "May I confirm if I'm speaking with…?").

## D-007 — Voice: the end of a turn could overtake its last transcript

**Found:** 2026-10-27, Day 5, by an offline test of `VoiceBridge`, before any live voice run.

**What happened.** The bridge turned a user turn into one debtor line: it collected the final
transcripts and flushed them on `UserStoppedSpeakingFrame`. In Pipecat that frame is a system
frame, and a processor handles system frames ahead of the data frames still in its queue
(`TranscriptionFrame` is a data frame). So the turn's end could be handled before its last
transcript: "Yes, it's me. Born August 8, 1993." reached the agent as "Yes, it's me.", and the
date became the start of the next debtor line.

- In a live call this would split one answer in two. Identity reads dates only from the
  debtor's own words since the last attempt, so a date carried into the next line could be
  missed or checked one turn late.

**How it was found.** The test sent two final transcripts and the turn end back to back, as
the STT and turn processor do within milliseconds. It got only the first transcript.

**Fix (code).** On `UserStoppedSpeakingFrame` the bridge queues its own end-of-turn marker (a
data frame) into its input queue. The marker is handled after every transcript already
queued, and the turn is flushed then. A turn that ends before any transcript still waits for
the next final transcript.

**Tests.**
- `tests/test_voice_offline.py::test_one_user_turn_becomes_one_debtor_line`
- `tests/test_voice_offline.py::test_a_late_final_transcript_still_ends_the_turn`

**Lesson.** In Pipecat, frame order across frame types is not arrival order: system frames
jump the queue. Any processor that pairs a system frame with data frames sent before it must
re-queue the system event behind them.
