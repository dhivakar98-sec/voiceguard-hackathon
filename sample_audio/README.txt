sample_audio/ — demo clips
==========================

Anything you drop in this folder shows up as a one-click "Try a sample" button
in the web UI (the backend lists the folder at /api/samples). Supported:
wav, flac, ogg, mp3 out of the box; m4a/aac need ffmpeg.

What is shipped here
--------------------
  ai_tts_english.wav          synthetic speech, macOS system TTS (US English)
  ai_tts_indian_english.wav   synthetic speech, macOS system TTS (Indian English)
  tone_not_speech.wav         a swept tone — not speech at all, use it to check
                              that the app degrades gracefully instead of
                              pretending to have an opinion

These are real synthetic-speech samples, but they are SYSTEM TTS, not modern
voice cloning. They are here so the app has something to chew on out of the box.

What you should add before the demo (this is the part that wins the room)
------------------------------------------------------------------------
  1. human_<name>.wav   — record a teammate on a phone, 8-15 seconds of speech.
  2. clone_<name>.wav   — clone that same teammate with any free voice-cloning
                          tool and have it read the same sentence.

Then show them side by side: real voice passes, the clone gets flagged. Also try
a phone-quality version (record the playback over a speakerphone) — that is the
condition where most commercial detectors fall over.

Keep clips short (under ~15 s) so the repo stays small enough to zip and email.
