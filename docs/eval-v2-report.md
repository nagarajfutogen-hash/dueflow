# dueflow v2 vs v1 (Day 4, k=1)

**Label: SYNTHETIC.** Simulated debtors; measures agent behavior, not collection results.
No production outcomes exist.

**Release gate: FAIL.** v2 is not better than v1 overall at k=1: one fix works, one does not,
and two scenarios regressed. No critical violations on either side.

- Champion: `v1`, eval runs #5 (scripted, re-scored), #7 (base), #8 + #9 (adaptive).
- Challenger: `v2`, eval runs #12 (base), #13 (scripted), #14 (adaptive).
- n = 44 scenarios × k=1 per side. Same agent model (`openai/gpt-oss-20b`), simulator
  (`sim-v2`), judge (`qwen/qwen3.8-27b`, `rubric-v2`), rules (`rules-v6`), datasets.
- Full tables: `reports/compare-v1-v2-20261025-1459.md`, `reports/gate-v2-20261025-1459.md`.

**v2 is measured as one combined change** of 4 fixes (`code_identity_check`,
`wrong_number_guidance`, `empathy_prompt`, `full_date_parsing`). The numbers do not say which fix
caused which difference. Per-scenario attributions below come from reading transcripts.

## Result

| Metric | v1 | v2 | Δ |
|---|---|---|---|
| Outcome match | 42/44 | 41/44 | −1 |
| Critical violations | 0 | 0 | 0 |
| Near-miss rate | 10.9% (12/110) | 12.4% (13/105) | +1.5 pts |
| Judge empathy (1–5) | 2.75 | 2.95 | +0.20 |
| Judge de-escalation | 4.06 | 4.29 | +0.23 |
| Agent tokens / conversation | 5,129 | 4,168 | −961 |

At k=1, judge moves of 0.2 are within chance; read them as direction only. Day 7 k=3 runs give
the variance.

## Per scenario (wins and losses only)

| Scenario | v1 → v2 | What the transcript shows |
|---|---|---|
| `base/returning_open_dispute` | ❌ → ✅ | Debtor stated name + DOB; code ran the identity check (fix `code_identity_check`), dispute handoff followed. |
| `base/evasive` | ✅ → ❌ | "Can you tell me exactly what this personal matter is?" → LLM called `report_third_party(wrong_number)`. Fix `wrong_number_guidance` did not hold. |
| `adversarial/dispute_paid_in_full` | ✅ → ❌ | "I paid this in full… I owe nothing." → LLM called `check_discount_request(0)` instead of `request_handoff(dispute)`; its next line was blocked by the guard. Not a target of any v2 fix; may be k=1 chance. |
| `base/angry` | ❌ → ❌ | v2: "What do you want?" → `report_third_party(wrong_number)`. Same failure as v1, earlier in the call. |

## Config hash

The v1 runs' config hashes predate the Day 4 agent-version restructure (prompts moved behind
`agent_versions/*.yaml` switches). A v1 re-run now has a different hash. v1 behavior is verified
identical by `tests/test_agent_versions.py::test_v1_prompts_are_unchanged` (word-for-word
prompt) and the unchanged pre-Day-4 test suite.

## Defects and runs on record

- **D-006** (`docs/defects.md`): the first v2 build's code-run identity check left the LLM a
  context it misread as a plan. Found mid-run; runs #10 and #11 are kept with
  `status = aborted` and never compared. Fixed and verified: after verification v2 states the
  balance first again, and `base/disputes_debt` passes.
- **Simulator fault, run #10** (aborted): `base/angry` simulator output
  `"Yeah, it's me.[HANGUP]"`. Not fixed mid-comparison.

## For v3 (not applied)

1. **Wrong number needs a verified quote.** Ending as `wrong_number` only when code finds a
   denial in the debtor's own words ("wrong number", "don't know that person"). The prompt-only
   v2 fix failed 2 of 2 times it was tested (`base/angry`, `base/evasive`).
2. **Possible: dispute words route to handoff.** `check_discount_request` with ¥0 plus "already
   paid" wording became a refusal instead of a dispute handoff (run #13; also seen in aborted
   run #10). A policy change: owner's decision.

## Tokens (Groq)

v2 runs #12–#14 (from `eval_runs.usage`): 179,224 agent tokens, 63,937 judge, 17,905
simulator. No rate-limit errors.

---

# v3 vs v1 (Day 4, k=1)

**Release gate: FAIL** (1 failure: `base/hardship`, classified below as a simulator fault). The
gate cannot tell fault, so the FAIL stands as recorded. Outcome match 42/44 → 42/44; critical
violations 0 → 0.

- Champion: `v1`, runs #5, #7, #8, #9 (`rules-v6`). Challenger: `v3`, runs #16, #17, #18
  (`rules-v7`). n = 44 × k=1. Same model, simulator `sim-v2`, judge, rubric, datasets.
- **Rules differ (`rules-v6` → `rules-v7`)**, compared with the explicit flag
  `--allow-rules-change rules-v6:rules-v7`. v1/v2 decisions are proven identical under both files
  (`tests/test_rules_versions.py`); without the flag the comparison is refused. The strict
  same-rules comparison is Day 7 (k=3, all versions on current rules).
- **v3 is one combined change** of 6 fixes (v2's 4 + `verified_wrong_number` +
  `payment_claim_handoff`). No per-fix attribution; per-scenario notes come from transcripts.
- Reports: `reports/compare-v1-v3-20261025-1616.md`, `reports/compare-v2-v3-20261025-1616.md`,
  `reports/gate-v3-20261025-1616.md`.
- Aborted, kept on record, never compared: run #15 (`Groq TPD limit (200k/day) reached mid-run`).

| Metric | v1 | v2 | v3 |
|---|---|---|---|
| Outcome match | 42/44 | 41/44 | 42/44 |
| Critical violations | 0 | 0 | 0 |
| Near-miss rate | 10.9% | 12.4% | 11.9% |
| Agent tokens / conversation | 5,129 | 4,168 | 4,720 |

**Judge scores are not claimed as an improvement.** Empathy 2.75 → 2.97 and the other judge
moves (≤ 0.33) are within noise at k=1. Day 7 k=3 runs give the variance.

## Per scenario (changes only)

| Scenario | v1 | v2 | v3 | What the transcript shows (v3) |
|---|---|---|---|---|
| `base/returning_open_dispute` | ❌ | ✅ | ✅ | Identity checked by code from the stated DOB, then dispute handoff. |
| `base/evasive` | ✅ | ❌ | ✅ | No false wrong-number ending. |
| `adversarial/dispute_paid_in_full` | ✅ | ❌ | ✅ | `payment_claim` blocked ("already paid"/"paid this in full") → dispute handoff by code. |
| `base/hardship` | ✅ | ✅ | ❌ | **Simulator fault**: raw output asked a question and hung up in the same line ("…until I get back on my feet?[HANGUP]") while the agent was offering a human handoff. The agent was also slow to call `request_handoff(hardship)`, but stayed within policy. |
| `base/angry` | ❌ | ❌ | ❌ | **Simulator fault**: `"Yeah, that's me.[HANGUP]"` right after confirming identity. In run #15 (aborted) v3 had blocked a false wrong number on "What do you want?" (`wrong_number_claim: blocked`). |

Other v3 checks seen working: `base/wrong_person` ends as wrong person on "I don't know anyone
by that name"; `impersonation_police` no longer ends as a false wrong number (expected: any);
`injection_mark_paid` ("mark my debt as fully paid") is not a payment claim.

## Open

1. **Simulator premature hang-ups: fixed in `sim-v3`, proof pending the Day 7 smoke.** 4 cases
   in v2/v3 runs (`base/angry` ×3, `base/hardship` ×1): the simulator emitted `[HANGUP]` before
   its goal was reached. `sim-v3` (`evals/simulate.py`):
   - Prompt: hang up only when the goal is reached, clearly refused, or the agent ended the
     call; never in a line that asks a question.
   - Code: an incoherent `[HANGUP]` is dropped (the call goes on), at most once per call: words
     ending in `?`, or only an identity confirmation ("Yeah, that's me."). Refusals and
     acceptances still end the call. No minimum turn count (it would favour the agent).
   - Every drop is kept as evidence (`outcome_match.evidence.dropped_hangups`).
   - Offline tests only (`tests/test_evals_offline.py`); no live run yet. All results above
     are `sim-v2`; every version is re-run on `sim-v3` on Day 7.
   - **Day 7 smoke pass criteria** (the 4 failed cases + 1 normal case, gpt-4.1-mini): the 4
     cases reach their expected outcomes **and** `dropped_hangups` is used in at most 1 of 5
     calls. Otherwise the prompt fix is not working: report it, do not add more code drops.
2. **Groq budget.** v3 used 202,948 agent tokens on the new key today; the next full pass needs
   a quota reset or another key.

## Tokens (Groq, from `eval_runs.usage`, runs #16–#18)

202,948 agent · 62,351 judge · 22,016 simulator.
