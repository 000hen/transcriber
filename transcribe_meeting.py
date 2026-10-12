import json
from pathlib import Path

import numpy as np
import torch
import whisper

torch.set_num_threads(4)
out = Path("transcripts")
out.mkdir(exist_ok=True)
model = whisper.load_model("turbo", download_root="models")
for path in sorted(Path(".").glob("*.aac")):
    target = out / (path.stem + ".json")
    if target.exists():
        continue
    print("START", path.name, flush=True)
    audio = whisper.load_audio(str(path))
    segments = []
    for offset in range(0, len(audio), 16000 * 120):
        chunk = audio[offset : offset + 16000 * 120]
        if np.max(np.abs(chunk)) < 0.001:
            continue
        result = model.transcribe(
            chunk,
            temperature=0.2,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            language="zh",
            task="transcribe",
            initial_prompt="以下是臺灣使用者的會議對話，請使用繁體中文。",
            condition_on_previous_text=False,
            verbose=None,
        )
        for s in result["segments"]:
            s["start"] += offset / 16000
            s["end"] += offset / 16000
            segments.append(s)
        (out / (path.stem + ".partial.json")).write_text(
            json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"PROGRESS {path.stem} {min((offset + len(chunk)) / 16000, len(audio) / 16000):.0f}s",
            flush=True,
        )
    target.write_text(
        json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        f"[{int(s['start']) // 60:02d}:{int(s['start']) % 60:02d}] {s['text']}"
        for s in segments
    ]
    (out / (path.stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
    print("DONE", path.name, flush=True)
