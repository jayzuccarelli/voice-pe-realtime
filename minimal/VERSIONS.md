# The two relays that have served the puck

Both run as the user unit `voicepe-minimal.service` on :8765 from
`~/.local/share/voicepe-minimal/relay.py`. The `VERSION` constant in the
file and the first log line say which one is running.

## live (2026-09-27 to 2026-11-04)

`minimal/relay.py` up to commit `e5b2559`. gpt-live-1 hears the puck's
audio, decides, delegates tool calls to its backend text model, and speaks.
A backup transcriber (gpt-4o-transcribe) reads any utterance Live produced
no transcript for and hands the words to the same backend, behind a
confidence filter.

Why it was replaced: Live's mouth speaks before its brain has acted ("Sure,
I'll take care of that", then nothing), its own hearing of a request is
rougher than a transcriber's, and nothing checks what was said against
what the house did. Measured on 44 recorded wakes (Oct 20-29): 5 of 14
commands done, 5 confirmations of actions that never ran. The confidence
filter also threw away correctly transcribed sentences on the strength of
one unusual word ("DJ Matrix").

Kept beside the deploy as `relay.py.bak_live_e5b2559`. To go back:

    cp ~/.local/share/voicepe-minimal/relay.py.bak_live_e5b2559 ~/.local/share/voicepe-minimal/relay.py
    systemctl --user restart voicepe-minimal

## chain (from 2026-11-04)

The chained design from OpenAI's voice-agent guide, nothing clever between
the stages:

1. wake word on the puck (unchanged firmware);
2. the production cutter turns the mic into utterances; two transcribers
   (gpt-transcribe, gpt-4o-mini-transcribe) read each one and the
   better-scoring English reading wins;
3. a text model (gpt-5.4-mini) with Home Assistant's tools acts; "On it."
   plays the moment a command is read, before the action;
4. what is spoken about an action is built from the tool result, never from
   the model's words; answers to questions are the model's text; a
   text-to-speech model (gpt-4o-mini-tts) voices it.

Household memory (HA's Memory API through the same MCP server) is carried
over from the live version: its instructions and index go into the model's
prompt, refreshed after a memory write.

Measured on the same 44 recordings before installing (evals/cascade_harness.py):
11 of 16 commands done, 0 invented confirmations, 13 of 13 questions
answered. The 5 misses: two loud-TV recordings, one drowned by room
conversation, one dead microphone, one self-correction ("the TV, uh, the
light").

What it gives up against live: Live's voice and sub-second replies.
Interruptions were never possible on this puck (the firmware closes the mic
while audio plays), so nothing is lost there.
