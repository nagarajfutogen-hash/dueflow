# Voice (Day 5)

Browser WebRTC → Silero VAD → Deepgram STT (`nova-3`, smart_format) → the same agent as text
(Flows, tools, policy checks, audit) → Deepgram TTS (`aura-2-thalia-en`). All voice results
are HUMAN TEST. Code: `app/voice/pipeline.py`, `app/api/voice.py`, dev page `/dev/voice.html`.

## How it works

- **Turns.** Pipecat VAD (default `VADParams`, stop after 0.2 s of silence) plus a 0.6 s
  speech timeout end the debtor's turn. Smart Turn is not used. A turn's final transcripts
  become one debtor line.
- **Guard.** Every spoken sentence is checked before TTS; a blocked sentence ends the reply
  early. Sentences are split exactly where the text output check splits a reply, so voice
  says what text would send (`test_per_sentence_check_says_what_the_whole_reply_check_says`).
  Each sentence writes its own `output_check` event, so voice near-miss rates are per sentence
  and are not compared with text runs.
- **Scripted lines** (opening, identity retries, terms, endings) are spoken word for word.
- **Amounts** are read as yen: TTS gets "12,000 yen" for "¥12,000" (`for_speech`). Voice
  only, after the guard: turns and events keep the guarded text exactly as checked.
- **Barge-in.** Pipecat stops the agent's audio when the debtor speaks over it. The agent
  turn's `latency_ms` gets `interrupted`, `interrupted_at_ms` and `spoken_chars`, and the
  LLM context is cut to what was heard (`trim_unspoken`).
- **Latency** on each agent turn's `latency_ms`: `ttfb_ms` = first TTS audio − VAD end of
  speech; `perceived_ms` = `ttfb_ms` + VAD stop wait. Report:
  `docker compose exec api python -m app.voice.latency --site vps` (P50/P95, nearest rank,
  with n). Official numbers come from the VPS; local numbers are labeled `local`.

## Known gaps

1. **Spoken dates (identity).** Identity reads dates from the debtor's own words. Deepgram's
   smart_format should write "August eighth nineteen ninety-three" as "August 8, 1993",
   which is read. Unformatted words and ambiguous forms ("08/08/1993") are not read: the
   agent asks again and the attempt is not counted. Offline tests cover the STT-formatted
   shapes only; that Deepgram produces them is proven live on the VPS only.
2. **Barge-in accuracy.** `spoken_chars` counts the TTS text frames that reached the output
   before the interruption. If Deepgram TTS sends no word timestamps, a whole sentence counts
   as heard once its audio starts, so `spoken_chars` (and the trimmed context) can include
   the rest of the sentence that was cut off. Checked live on the VPS.
3. **No measured latency yet.** Nothing has been measured locally (WebRTC does not reach the
   Docker Desktop container). The first numbers come from the VPS run.
