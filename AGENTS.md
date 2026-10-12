# Agent instructions

## Scope and workflow

- Keep README.md limited to description, installation, usage, and license.
- Maintain agent guidance in AGENTS.md. CLAUDE.md is a relative symbolic link to
  this file; do not replace it with an independent copy.
- Preserve existing recordings, transcripts, minutes, model caches, credentials,
  and unrelated work. Do not run paid API requests without user authorization.
- Keep the legacy `transcribe_meeting.py` separate from the standalone
  `meeting_summary.py` pipeline unless the task explicitly involves it.
- Never commit `.env`, API keys, or private meeting artifacts.
- Preserve source evidence and uncertainty: do not invent speakers, meeting
  dates, owners, deadlines, or decisions.

## Code discovery

Prefer codebase-memory-mcp graph tools for code discovery, in this order:
`search_graph`, `trace_path`, `get_code_snippet`, `query_graph`, and
`get_architecture`. Fall back to text/file searches for string literals,
configuration, non-code files, or insufficient graph results.

## Multi-agent workflow

Direct work is the default for small tasks. Delegate only when independent work
or context isolation materially improves correctness, speed, or review quality.
Use a read-only planner for substantial design risk, an implementer with explicit
non-overlapping ownership, and an independent reviewer when regression,
compatibility, concurrency, or security risk warrants it. Subagents must not
spawn further agents. The main agent owns integration, combined diff review,
relevant validation, and reporting unverified behavior and remaining risks.
Preserve other contributors' edits.

## Validation

Run the relevant tests after code changes:

```powershell
.\.venv\Scripts\python.exe -m unittest -v
```

API behavior is mocked; tests need no credentials and incur no API charges.
Report exactly what was verified, and distinguish mocked tests from live API runs.
Documentation-only changes need link and diff checks, not an API run.

## Pipeline reference

The following implementation and operational notes were moved from README.md.
Keep them aligned with the code when behavior changes.

## Setup (Windows PowerShell)

Requires Python 3.11+ and FFmpeg on `PATH`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If virtualenv creation reports an `ensurepip` error but creates its interpreter,
install through the system pip instead:

```powershell
python -m pip --python .venv install -r requirements.txt
```

Set `OPENAI_API_KEY` in a `.env` file beside `meeting_summary.py`, or in the process
environment. The script loads that `.env` even when launched from another folder;
existing environment variables take precedence. `.env` is ignored by Git.
No credential is needed for `--prepare-only`.

```dotenv
OPENAI_API_KEY=your-api-key
```

## Run

```powershell
# Local preparation only: no uploads or API requests.
.\.venv\Scripts\python.exe meeting_summary.py "*.aac" --prepare-only

# After adding OPENAI_API_KEY, run the complete pipeline.
.\.venv\Scripts\python.exe meeting_summary.py "*.aac"

# Optional speed-up; use normal speed first to assess transcription quality.
.\.venv\Scripts\python.exe meeting_summary.py "*.aac" --speed 1.25

# Explicit input order and timing offsets for tracks that start later.
.\.venv\Scripts\python.exe meeting_summary.py alice.wav bob.wav --offsets 0 12.5

# Lower-cost summary option.
.\.venv\Scripts\python.exe meeting_summary.py "*.aac" --summary-model gpt-6-luna

# Use Whisper when segment timestamps are needed.
.\.venv\Scripts\python.exe meeting_summary.py "*.aac" --transcription-model whisper-1
```

Quoted globs expand inside Python, sorted within each pattern. Explicit file order
is retained, and duplicate paths are removed. Put input files before options.
All inputs belong to the same meeting. They are assumed to share time zero unless
`--offsets` supplies one nonnegative start time per expanded input. Sequential
recordings need cumulative offsets. There is no automatic synchronization.

Each file is processed separately to retain overlapping speech and track labels;
tracks are not mixed into one waveform. Filenames label the sources, not verified
speaker identities. Shared microphones and audio bleed may produce duplicates;
the summary prompt asks the model to consolidate them, without guaranteed deduplication.

## Output and resume

Final results are written directly to `output/YYYY-MM-DD/`:

- `會議摘要.md`: meeting information, narrative core summary, numbered discussion
  sections with approximate source times, and a consensus/unresolved-items table.
- `待辦事項.md`: checkboxes, a questions table, and next-meeting details.
- `逐字稿.md`: the separate transcript targeted by evidence links in both documents.
- `minutes.json`, `transcript.json`, `configuration.json`: structured content and settings.

The folder date defaults to the processing date in Taiwan (UTC+8). It does **not**
become the meeting date in the document; unknown meeting dates stay unknown.
Use `--date 2026-10-09` to choose a folder date, and `--output other-folder` to
change the root. Example:

```powershell
.\.venv\Scripts\python.exe meeting_summary.py "*.aac" --date 2026-10-09
```

The latest successful run for a date replaces that date's published files.
Earlier successful results remain archived by content/settings hash. Use distinct
`--output` roots for separate meetings on the same date if both must remain published.
The formatting follows the example documents; it does not copy their meeting
facts, professor references, names, or deadlines into unrelated meetings.

Working artifacts and archives live under `output/.cache/<input-and-preparation-hash>/`:

- `manifest.json`: input hashes, offsets, VAD settings, duration statistics,
  compressed files, and mapping from retained audio to original sample ranges.
- `tNNN-cNNNNN.mp3`: inspect these when VAD or recognition appears inaccurate.
- `transcriptions/`: successful API responses saved after each chunk.
- `summaries/`: validated intermediate summaries, including long-meeting batches.
- `results/<transcript-and-summary-settings-hash>/transcript.json`: all tracks
  sorted by original meeting time, including source ranges and stable evidence IDs.
- `results/.../`: archived copies of the six published files above.
- `results/.../configuration.json`: model and language settings for this result.

Rerun the same command to reuse completed work. Changed audio or preparation
settings get a new directory; changed models/language/prompts invalidate their
respective request caches. An interrupted in-flight request may need to be sent
again and may have already been billed. Do not run two identical jobs concurrently.
If no speech/text is detected, the script stops without inventing minutes.
`--prepare-only` writes working artifacts only, without publishing meeting documents.
Caches from the previous `meeting-output/` location are preserved but not automatically
discovered; copy their hash-named directories into `output/.cache/` to reuse audio
preparation and transcription. The new summary schema requires a new summary request.

## Accuracy and cost controls

- Default transcription is `gpt-transcribe`, using language hints and coarse
  chunk-level timing. Select `whisper-1` for segment timestamps. Timestamps are
  estimates; `source_ranges` records gaps removed within a segment.
- Default summary model is `gpt-6.1-sol`; use `--summary-model gpt-6-luna` for
  the lower-cost option, or override with `--summary-model` using a
  model available to your project that supports Responses structured outputs.
- `--vad-mode 0` is less aggressive; `3` is most aggressive. The default is `1`,
  with `--padding 0.3` around speech. Use `--no-vad` if quiet speech gets dropped.
- Speech islands are packed in source order into chunks of at most 120 retained
  seconds, preferring gaps between utterances. Very long speech can be split at a chunk boundary. Padding protects
  VAD boundaries but does not restore context across chunks.
- `--speed` defaults to `1.0` and accepts up to `2.0`. Faster audio can reduce
  clarity. MP3 compression reduces bytes; it does not itself shorten duration.
  The printed durations are **not price estimates or a billing guarantee**.
- `--summary-chars 24000` bounds each text batch by characters, not tokens. Long
  meetings use intermediate summaries before synthesis. No transcript is silently
  truncated; if reduction cannot fit, the script stops and preserves intermediates.
  Increase the limit only within the selected model's context capacity.
- Every summary item must cite existing transcript IDs. Unknown owners/deadlines
  remain null / 待確認. Valid citation IDs and a stable schema do not prove factual
  accuracy; review the evidence and original audio before relying on decisions.

Running without `--prepare-only` uploads the prepared audio and transcript text
to OpenAI. The API key is read only at runtime. API calls have bounded retries;
completed responses are cached. Responses summary calls use `store=False`.

## Validation

```powershell
.\.venv\Scripts\python.exe -m unittest -v
```

Tests exercise VAD silence handling, timing restoration after silence removal and
speed-up, multi-track ordering, cache invalidation, summary source validation,
long-meeting reduction, refusal handling, missing credentials, and real FFmpeg
preparation. API behavior is mocked; no test requires a key or incurs API charges.

API contracts: [OpenAI file transcription](https://developers.openai.com/api/docs/guides/speech-to-text)
and [structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
