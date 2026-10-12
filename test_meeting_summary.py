import json
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import meeting_summary as m


class PipelineTests(unittest.TestCase):
    def test_gpt_transcribe_request_and_chunk_timing(self):
        args = m.parser().parse_args(["track.wav"])
        self.assertEqual(args.transcription_model, "gpt-transcribe")
        self.assertEqual(args.summary_model, "gpt-6.1-sol")
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / "audio.mp3").write_bytes(b"test")
            chunk = {
                "id": "t1",
                "file": "audio.mp3",
                "sha256": "test",
                "track": "Alice",
                "pieces": m.pack_spans([(0, 32000)], 32000)[0],
                "speed": 1,
                "offset": 5,
            }
            client = Mock()
            client.audio.transcriptions.create.return_value.model_dump.return_value = {
                "text": "會議內容",
                "languages": [{"code": "zh"}],
            }
            entries = m.transcribe(
                client,
                run,
                {"chunks": [chunk]},
                args.transcription_model,
                args.language,
            )
            request = client.audio.transcriptions.create.call_args.kwargs
            self.assertEqual(request["model"], "gpt-transcribe")
            self.assertEqual(request["extra_body"], {"languages": ["zh"]})
            self.assertNotIn("language", request)
            self.assertNotIn("timestamp_granularities", request)
            self.assertNotIn("response_format", request)
            self.assertEqual(entries[0]["timing"], "chunk")
            self.assertEqual((entries[0]["start"], entries[0]["end"]), (5, 7))

    def test_short_utterance_is_not_split_at_chunk_boundary(self):
        groups = m.pack_spans([(0, 90), (200, 240)], 100)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[1][0]["source_start"], 200)
        self.assertEqual(groups[1][0]["source_end"], 240)

    def test_pack_preserves_every_sample_and_chunk_limit(self):
        spans = [(10, 100), (200, 250), (400, 601)]
        groups = m.pack_spans(spans, 100)
        actual = [
            n
            for g in groups
            for p in g
            for n in range(p["source_start"], p["source_end"])
        ]
        self.assertEqual(actual, [n for a, b in spans for n in range(a, b)])
        self.assertTrue(all(g[-1]["packed_end"] <= 100 for g in groups))
        self.assertTrue(all(g[0]["packed_start"] == 0 for g in groups))

    def test_timing_restores_removed_silence_speed_and_offset(self):
        pieces = m.pack_spans([(0, 16000), (160000, 192000)], 100000)[0]
        self.assertEqual(
            m.source_ranges(pieces, 0.25, 1, 2, 5), [[5.5, 6.0], [15.0, 16.0]]
        )
        self.assertEqual(m.source_ranges(pieces, 0.5, 1.5, 2, 0), [[10.0, 12.0]])

    def test_vad_silence_and_partial_final_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "silent.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setparams((1, 2, m.RATE, 0, "NONE", "not compressed"))
                wav.writeframes(b"\x00\x00" * 16001)
            self.assertEqual(m.intervals(path, 1, 0.3, False), ([], 16001))
            self.assertEqual(m.intervals(path, 1, 0.3, True), ([(0, 16001)], 16001))

    def test_transcribe_merges_tracks_and_reuses_completed_requests(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            chunks = []
            for name, offset in [("later", 50), ("earlier", 0)]:
                (run / name).write_bytes(name.encode())
                chunks.append(
                    {
                        "id": name,
                        "file": name,
                        "sha256": name,
                        "track": name,
                        "pieces": m.pack_spans([(0, 32000)], 32000)[0],
                        "speed": 2,
                        "offset": offset,
                    }
                )
            client = Mock()
            client.audio.transcriptions.create.return_value.model_dump.return_value = {
                "segments": [{"start": 0, "end": 1, "text": "討論"}]
            }
            first = m.transcribe(client, run, {"chunks": chunks}, "whisper-1", "zh")
            self.assertEqual([e["track"] for e in first], ["earlier", "later"])
            self.assertEqual(first[1]["end"], 52)
            self.assertEqual(
                m.transcribe(client, run, {"chunks": chunks}, "whisper-1", "zh"), first
            )
            self.assertEqual(client.audio.transcriptions.create.call_count, 2)
            m.transcribe(client, run, {"chunks": chunks}, "whisper-1", "en")
            self.assertEqual(client.audio.transcriptions.create.call_count, 4)

    def test_summary_validates_sources_and_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            client = Mock()
            valid = m.Minutes(
                summary=[m.Record(text="決定測試", sources=["s1"])],
                decisions=[],
                open_questions=[],
                action_items=[],
            )
            client.responses.parse.return_value = SimpleNamespace(
                status="completed", output_parsed=valid
            )
            entries = [{"id": "s1", "text": "決定測試"}]
            self.assertEqual(m.summarize(client, entries, run, "model", 4000), valid)
            self.assertEqual(m.summarize(client, entries, run, "model", 4000), valid)
            self.assertEqual(client.responses.parse.call_count, 1)
            invalid = valid.model_copy(deep=True)
            invalid.summary[0].sources = ["invented"]
            client.responses.parse.return_value.output_parsed = invalid
            with self.assertRaisesRegex(ValueError, "unknown evidence"):
                m.summarize(client, entries, run, "different-model", 4000)

    def test_failed_second_request_resumes_from_first_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            chunks = []
            for number in range(2):
                name = str(number)
                (run / name).write_bytes(name.encode())
                chunks.append(
                    {
                        "id": name,
                        "file": name,
                        "sha256": name,
                        "track": name,
                        "pieces": m.pack_spans([(0, 32000)], 32000)[0],
                        "speed": 1,
                        "offset": 0,
                    }
                )
            response = Mock()
            response.model_dump.return_value = {"text": "完整句子"}
            client = Mock()
            client.audio.transcriptions.create.side_effect = [
                response,
                RuntimeError("network"),
            ]
            with self.assertRaisesRegex(RuntimeError, "network"):
                m.transcribe(
                    client, run, {"chunks": chunks}, "gpt-4o-mini-transcribe", "zh"
                )
            client.audio.transcriptions.create.side_effect = [response]
            entries = m.transcribe(
                client, run, {"chunks": chunks}, "gpt-4o-mini-transcribe", "zh"
            )
            self.assertEqual(len(entries), 2)
            self.assertTrue(all(e["timing"] == "chunk" for e in entries))
            self.assertEqual(client.audio.transcriptions.create.call_count, 3)

    def test_markdown_evidence_links_target_transcript_headings(self):
        minutes = m.Minutes(
            summary=[m.Record(text="重點", sources=["t001-c00001-s0001"])],
            decisions=[],
            open_questions=[],
            action_items=[],
        )
        content = m.render(
            minutes,
            [
                {
                    "id": "t001-c00001-s0001",
                    "track": "Alice",
                    "start": 0,
                    "end": 1,
                    "timing": "segment",
                    "text": "內容",
                }
            ],
        )
        self.assertIn("](逐字稿.md#t001-c00001-s0001)", content)
        transcript = m.render_transcript(
            [
                {
                    "id": "t001-c00001-s0001",
                    "track": "Alice",
                    "start": 0,
                    "end": 1,
                    "timing": "segment",
                    "text": "內容",
                }
            ]
        )
        self.assertIn("### t001-c00001-s0001", transcript)

    def test_reference_format_and_dated_publication(self):
        minutes = m.Minutes(
            title="研究討論",
            summary=[m.Record(text="核心內容", sources=["s"])],
            decisions=[m.Record(title="方向", text="確認方案", sources=["s"])],
            open_questions=[
                m.Record(title="模型 | 選擇", text="尚待\n確認", sources=["s"])
            ],
            action_items=[
                m.Action(
                    title="整理資料",
                    text="核對文獻",
                    sources=["s"],
                    owner=None,
                    due=None,
                )
            ],
            discussions=[
                m.Record(title="研究流程", text="完整討論段落", sources=["s"])
            ],
            next_meeting=[
                m.Record(title="時間", text="星期一晚上，日期待確認", sources=["s"])
            ],
        )
        entries = [
            {
                "id": "s",
                "track": "Alice",
                "start": 3,
                "end": 20,
                "timing": "chunk",
                "text": "原文",
            }
        ]
        summary = m.render(minutes, entries)
        actions = m.render_actions(minutes)
        self.assertIn("# 研究討論｜會議摘要", summary)
        self.assertIn("### 1. 研究流程（約 00:00:03–00:00:20）", summary)
        self.assertIn("會議日期**：錄音中未明確交代", summary)
        self.assertIn("| 模型 \\| 選擇 |", summary)
        self.assertIn("- [ ] **整理資料**：核對文獻", actions)
        self.assertIn("負責人：待確認；期限：待確認", actions)
        self.assertIn("## 下次會議", actions)
        self.assertNotIn("### s", summary)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = root / "archive"
            result.mkdir()
            files = {
                "會議摘要.md": summary,
                "待辦事項.md": actions,
                "逐字稿.md": m.render_transcript(entries),
                "minutes.json": "{}",
                "transcript.json": "[]",
                "configuration.json": "{}",
            }
            for name, content in files.items():
                (result / name).write_text(content, encoding="utf-8")
            output = m.publish_results(result, root / "output", m.date(2026, 10, 12))
            self.assertEqual(output, root / "output" / "2026-10-12")
            self.assertEqual(
                (output / "會議摘要.md").read_text(encoding="utf-8"), summary
            )
            self.assertEqual(set(p.name for p in output.iterdir()), set(files))
            self.assertTrue((result / "會議摘要.md").exists())

    def test_new_sections_validate_evidence(self):
        minutes = m.Minutes(
            summary=[],
            decisions=[],
            open_questions=[],
            action_items=[],
            discussions=[m.Record(text="議題", sources=["unknown"])],
        )
        with self.assertRaisesRegex(ValueError, "unknown evidence"):
            m.check_sources(minutes, {"known"})

    def test_recursive_summary_includes_every_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            client = Mock()

            def respond(**kwargs):
                data = json.loads(kwargs["input"][1]["content"])
                ids = (
                    [x["id"] for x in data]
                    if "id" in data[0]
                    else [
                        source
                        for note in data
                        for item in note["summary"]
                        for source in item["sources"]
                    ]
                )
                return SimpleNamespace(
                    status="completed",
                    output_parsed=m.Minutes(
                        summary=[m.Record(text="摘要", sources=ids)],
                        decisions=[],
                        open_questions=[],
                        action_items=[],
                    ),
                )

            client.responses.parse.side_effect = respond
            entries = [{"id": str(i), "text": "x" * 2300} for i in range(3)]
            result = m.summarize(client, entries, Path(folder), "model", 4000)
            self.assertEqual(result.summary[0].sources, ["0", "1", "2"])
            self.assertEqual(client.responses.parse.call_count, 4)

    def test_refusal_leaves_no_cached_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            client = Mock()
            client.responses.parse.return_value = SimpleNamespace(
                status="completed", output_parsed=None
            )
            with self.assertRaisesRegex(RuntimeError, "refused or incomplete"):
                m.summarize(
                    client, [{"id": "s", "text": "hi"}], Path(folder), "model", 4000
                )
            self.assertFalse((Path(folder) / "summaries").exists())

    def test_success_on_last_allowed_reduction_is_returned(self):
        with tempfile.TemporaryDirectory() as folder:
            client = Mock()

            def respond(**kwargs):
                data = json.loads(kwargs["input"][1]["content"])
                source = (
                    data[0]["id"]
                    if "id" in data[0]
                    else data[0]["summary"][0]["sources"][0]
                )
                return SimpleNamespace(
                    status="completed",
                    output_parsed=m.Minutes(
                        summary=[m.Record(text="x" * 1600, sources=[source])],
                        decisions=[],
                        open_questions=[],
                        action_items=[],
                    ),
                )

            client.responses.parse.side_effect = respond
            entries = [{"id": str(i), "text": "x" * 1600} for i in range(512)]
            result = m.summarize(client, entries, Path(folder), "model", 4000)
            self.assertEqual(result.summary[0].sources, ["0"])
            self.assertEqual(client.responses.parse.call_count, 511)

    def test_no_key_fails_before_preparation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "track.wav"
            path.touch()
            with (
                patch.dict(m.os.environ, {"OPENAI_API_KEY": ""}),
                patch.object(m, "prepare") as prepare,
            ):
                with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                    m.main([str(path)])
                prepare.assert_not_called()

    def test_local_ffmpeg_preparation_and_cached_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "track.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setparams((1, 2, m.RATE, 0, "NONE", "not compressed"))
                wav.writeframes(b"\x00\x00" * 32000)
            args = m.parser().parse_args([str(path), "--no-vad", "--speed", "1.25"])
            run, manifest = m.prepare([path], [3], args, root / "out")
            self.assertEqual(len(manifest["chunks"]), 1)
            self.assertEqual(manifest["stats"][0]["upload_seconds_approx"], 1.6)
            self.assertGreater((run / manifest["chunks"][0]["file"]).stat().st_size, 0)
            with patch.object(
                m, "ffmpeg", side_effect=AssertionError("Cache was not reused")
            ):
                self.assertEqual(
                    m.prepare([path], [3], args, root / "out"), (run, manifest)
                )


if __name__ == "__main__":
    unittest.main()
