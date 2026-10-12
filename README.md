# Meeting audio → minutes

A Python CLI that prepares one or more audio tracks with local voice activity
detection, transcribes them through OpenAI, and produces meeting minutes and
action items in Taiwan Traditional Chinese.

## Installation

Requires Python 3.11+ and FFmpeg on `PATH`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Set `OPENAI_API_KEY` in your environment or a `.env` file beside
`meeting_summary.py`.

## Usage

```powershell
# Prepare audio locally without API calls.
.\.venv\Scripts\python.exe meeting_summary.py "*.aac" --prepare-only

# Transcribe and summarize one meeting (uploads audio and text to OpenAI).
.\.venv\Scripts\python.exe meeting_summary.py "*.aac"

# Show all options.
.\.venv\Scripts\python.exe meeting_summary.py --help
```

Use tracks from the same meeting. For tracks starting at different times, pass
one offset in seconds per input, for example `alice.wav bob.wav --offsets 0 12.5`.

Results are saved to `output/YYYY-MM-DD/`, including `會議摘要.md`,
`待辦事項.md`, and `逐字稿.md`. Rerunning the same command reuses cached work;
the latest successful run replaces that date's published files. Use separate
`--output` folders for different meetings on the same date.

## License

This project is licensed under the [MIT License](LICENSE).
