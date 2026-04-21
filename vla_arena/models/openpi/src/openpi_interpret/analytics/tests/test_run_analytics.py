"""Integration tests for the analytics CLI pipeline."""

from __future__ import annotations

from pathlib import Path

import yaml
import pytest


class TestProcessEpisode:
    def test_episode_produces_metrics(self, analytics_reader) -> None:
        from analytics.run_analytics import process_episode

        result = process_episode(analytics_reader, layer=17)
        assert result.episode_id == "test_analytics_ep"
        assert result.num_timesteps == 2
        assert len(result.cmf_per_timestep) == 2
        assert len(result.silhouette_per_timestep) == 2

    def test_cmf_values_in_range(self, analytics_reader) -> None:
        from analytics.run_analytics import process_episode

        result = process_episode(analytics_reader, layer=17)
        for ts_cmf in result.cmf_per_timestep:
            for v in ts_cmf.as_dict().values():
                assert -1.0 <= v <= 1.0

    def test_silhouette_values_in_range(self, analytics_reader) -> None:
        from analytics.run_analytics import process_episode

        result = process_episode(analytics_reader, layer=17)
        for ts_sil in result.silhouette_per_timestep:
            assert -1.0 <= ts_sil.score <= 1.0

    def test_visual_attention_by_layer(self, analytics_reader) -> None:
        from analytics.run_analytics import process_episode

        result = process_episode(analytics_reader, layer=17)
        assert len(result.visual_attention_by_layer) == 7
        for lv in result.visual_attention_by_layer:
            assert lv.layer in [0, 3, 6, 9, 12, 15, 17]
            assert len(lv.head_scores) == 8
            for h in lv.head_scores:
                assert 0.0 <= h <= 1.0
            assert 0.0 <= lv.layer_mean <= 1.0

    def test_silhouette_by_layer(self, analytics_reader) -> None:
        from analytics.run_analytics import process_episode

        result = process_episode(analytics_reader, layer=17)
        assert len(result.silhouette_by_layer) == 7
        for ls in result.silhouette_by_layer:
            assert ls.layer in [0, 3, 6, 9, 12, 15, 17]
            assert -1.0 <= ls.score <= 1.0


class TestReportGeneration:
    def test_full_pipeline_writes_yaml(self, analytics_data_dir: Path, tmp_path: Path) -> None:
        from analytics.reader import scan_episodes
        from analytics.run_analytics import build_report, process_episode, report_to_dict, write_yaml

        readers = scan_episodes(analytics_data_dir)
        assert len(readers) == 1

        episodes = [process_episode(r, layer=17) for r in readers]
        report = build_report(episodes, layer=17)
        report_dict = report_to_dict(report)

        output_path = tmp_path / "test_report.yaml"
        write_yaml(report_dict, output_path)
        assert output_path.exists()

        with open(output_path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["metadata"]["layer"] == 17
        assert loaded["metadata"]["num_episodes"] == 1
        assert "global" in loaded
        assert "silhouette" in loaded["global"]
        assert "cmf" in loaded["global"]
        assert len(loaded["episodes"]) == 1

        ep = loaded["episodes"][0]
        assert "cmf" in ep
        assert set(ep["cmf"].keys()) == {"S_to_L", "S_to_V", "A_to_L", "A_to_V", "A_to_S"}
        assert "silhouette" in ep
        assert "mean" in ep["silhouette"]
        assert "per_timestep" in ep["silhouette"]

        assert "visual_attention_ranking" in loaded
        va = loaded["visual_attention_ranking"]
        assert "uniform_baseline" in va
        assert "global_mean" in va
        assert len(va["layers"]) == 7
        for entry in va["layers"]:
            assert "layer" in entry
            assert "layer_mean" in entry
            assert len(entry["heads"]) == 8
            for h in entry["heads"]:
                assert "head" in h
                assert "visual_share" in h
                assert isinstance(h["head"], int)
                assert 0 <= h["visual_share"] <= 1.0

        assert "silhouette_profile" in loaded
        sp = loaded["silhouette_profile"]
        assert len(sp["layers"]) == 7
