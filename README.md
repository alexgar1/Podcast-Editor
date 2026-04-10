# Podcast Editor

This first-pass tool:

- extracts only `HH:MM:SS:FF` timecode ranges from noisy GPT output
- opens a Premiere `.prproj` by gunzipping the XML payload
- falls back to the latest sibling auto-save if the main project is missing or empty
- leaves the source sequence untouched
- creates a new sibling sequence
- saves that new sequence back into the current project by default
- copies every overlapping video/audio layer from each selected timeline range into that new sequence
- can render the sequence audio by mixing overlapping audio tracks and re-transcribe it with the local Whisper helper
- writes a canonical base transcript with word-level timing and speaker IDs for downstream features
- writes a numbered speaker-block transcript for downstream LLM editing passes
- gzips the edited XML back into a Premiere project file

## End-To-End Automation

If you only have the Premiere project, use the automated entry point. It will:

- transcribe the source sequence with local Whisper
- convert that transcript into sentence-aligned Premiere-style `HH:MM:SS:FF` lines using Whisper word timings when available
- also write a numbered speaker-block transcript for broader editorial passes
- run a multi-pass Claude selection workflow that nominates candidate windows per transcript chunk and then reranks a shortlist in fresh context
- parse Claude's final chosen time ranges
- build a new trailer-selects sequence without touching the source synced sequence
- optionally run a broad concision Claude workflow that identifies redundant or off-topic ranges and creates a duplicate sequence with cut-and-lift gaps at those ranges

Run it from the `agent` conda env so Whisper is importable:

```bash
conda run -n agent python auto_podcast_editor.py \
  --project "/Users/alexgarrett/VIDEO/CLIENT/KAEDIM/PODCAST 10/Kaedim Podcast Episode 10.prproj" \
  --create-selects-sequence
```

That updates the existing project in place and writes:

- `<project>.analysis/<source-sequence>.wav`
- `<project>.analysis/<source-sequence>.json`
- `<project>.analysis/<source-sequence>.base_transcript.json`
- `<project>.analysis/<source-sequence>.editorial_blocks.txt`
- `<project>.analysis/<source-sequence>.timecoded.txt`
- `<project>.analysis/claude_prompt.txt`
- `<project>.analysis/claude_concision.prompt.md`
- `<project>.analysis/claude_response.txt`
- `<project>.analysis/claude_concision_response.txt`

If the project already contains both a source sequence and one or more generated `- Selects` sequences, the tool now
automatically uses the lone non-selects sequence. Pass `--sequence` when the project still has multiple plausible
source sequences.

If `ANTHROPIC_API_KEY` is not already exported, the script loads it from `.env`.
The editable Claude prompt files now live in:
- [claude_selects_prompt.md](/Users/alexgarrett/CODING/pcast-editor/claude_selects_prompt.md) for the shared editorial guidance
- [claude_first_pass_prompt.md](/Users/alexgarrett/CODING/pcast-editor/claude_first_pass_prompt.md) for the chunk-level nomination stage
- [claude_final_ranking_prompt.md](/Users/alexgarrett/CODING/pcast-editor/claude_final_ranking_prompt.md) for the shortlist reranking stage
- [claude_concision_prompt.md](/Users/alexgarrett/CODING/pcast-editor/claude_concision_prompt.md) for the full broad-concision prompt template

Use `--prompt-file` if you want to point the shared editorial guidance at a different file. The stage templates remain editable Markdown files in the repo, and the tool also writes the fully rendered per-call prompts into the analysis directory for inspection.
Use `--output-project` if you explicitly want to write the updated project to a different `.prproj`.

To run only the broad concision workflow:

```bash
conda run -n agent python auto_podcast_editor.py \
  --project "/Users/alexgarrett/VIDEO/CLIENT/KAEDIM/PODCAST 10/Kaedim Podcast Episode 10.prproj" \
  --create-concision-sequence
```

That creates a new sibling sequence named `<source sequence> - Concision` when Claude identifies removable ranges.
The concision sequence preserves the original timing and leaves timeline gaps where those ranges were lifted.
The concision workflow now sends the full numbered editorial-block transcript to Claude in one pass instead of chunking and merging partial decisions.
If no redundant or off-topic ranges are identified, the tool writes the analysis artifacts but does not create a concision sequence.

To run both workflows together:

```bash
conda run -n agent python auto_podcast_editor.py \
  --project "/Users/alexgarrett/VIDEO/CLIENT/KAEDIM/PODCAST 10/Kaedim Podcast Episode 10.prproj" \
  --create-selects-sequence \
  --create-concision-sequence
```

## Usage

```bash
python3 podcast_editor.py \
  --project "/Users/alexgarrett/VIDEO/CLIENT/KAEDIM/PODCAST 10/Kaedim Podcast Episode 10.prproj" \
  --notes-file "./fixtures/sample_gpt_output.txt"
```

That updates:

```text
/Users/alexgarrett/VIDEO/CLIENT/KAEDIM/PODCAST 10/Kaedim Podcast Episode 10.prproj
```

If the main project has no usable sequence, the tool automatically tries the newest file in `Adobe Premiere Pro Auto-Save/` and still writes the edited result back to the main project path by default.

## Sequence Transcription

The main editor can also render the selects sequence and transcribe it immediately:

```bash
python3 podcast_editor.py \
  --project "/Users/alexgarrett/VIDEO/CLIENT/KAEDIM/PODCAST 10/Kaedim Podcast Episode 10.prproj" \
  --notes-file "./fixtures/sample_gpt_output.txt" \
  --transcribe-selects \
  --whisper-model tiny.en
```

That writes transcript artifacts into:

```text
/Users/alexgarrett/VIDEO/CLIENT/KAEDIM/PODCAST 10/Kaedim Podcast Episode 10.transcript/
```

The transcription step shells out to `transcribe_sequence.py`. By default it runs through `conda run -n agent python ...`, but you can override that with `--whisper-python` or disable the conda wrapper with `--whisper-conda-env ""`.

The helper now always writes a canonical `<sequence>.base_transcript.json` artifact. That file is the detailed source of truth for future editing features and includes word-level timing, segment timing, speaker blocks, and a `speaker_id` on each word. The trailer-selection prompt still uses the same sentence-formatted `<sequence>.timecoded.txt` transcript as before, so the LLM input format is unchanged.
The new broad-concision workflow uses `<sequence>.editorial_blocks.txt`, which is a numbered speaker-block transcript with exact timecodes on each block.

Speaker detection is enabled by default on a best-effort basis.

The diarization option uses `pyannote/speaker-diarization-community-1`, which requires `pyannote.audio` plus a Hugging Face or pyannote token unless you point `--speaker-diarization-model` at a local downloaded pipeline. Export one of `PYANNOTE_AUTH_TOKEN`, `HUGGINGFACE_TOKEN`, or `HF_TOKEN`, or pass `--speaker-auth-token` directly. If speaker detection is not available, the base transcript still gets a stable `speaker_id` field, but it falls back to `speaker_unknown`. Pass `--no-detect-speakers` if you want to skip speaker detection entirely.

You can still run the transcription helper directly after the selects sequence exists:

```bash
conda run -n agent python transcribe_sequence.py \
  --project "/Users/alexgarrett/VIDEO/CLIENT/KAEDIM/PODCAST 10/Kaedim Podcast Episode 10.prproj" \
  --sequence "Kaedim Podcast Episode 10 - Selects"
```

That writes:

- `<sequence>.wav`
- `<sequence>.segments.json`
- `<sequence>.txt`
- `<sequence>.json`
- `<sequence>.base_transcript.json`
- `<sequence>.speaker_blocks.txt`
- `<sequence>.editorial_blocks.txt`
- `<sequence>.srt`
- `<sequence>.vtt`
- `<sequence>.tsv`

The raw Whisper JSON is still written and remains Whisper-compatible. The new base transcript JSON adds:

- `speaker_id` on each segment and timed word when speaker detection can align it
- `speaker_diarization` with the raw speaker turns used for alignment
- `speaker_blocks` with continuous same-speaker transcript blocks
- timeline-aligned ticks and `HH:MM:SS:FF` timecodes on each word and segment

This currently generates external transcript artifacts. It does not yet write transcript objects back into Premiere's internal transcript XML structures.

## Current Assumptions

- one unambiguous source sequence unless `--sequence` is provided
- nominal timecode math uses the rounded nominal fps for rates like 29.97/59.94
- no drop-frame handling yet
- the new sequence name defaults to `<source sequence> - Selects`
- the broad concision sequence name defaults to `<source sequence> - Concision`
