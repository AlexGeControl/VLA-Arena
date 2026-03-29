"""Tests for the attention router (C4): row + breakdown, summary."""

from __future__ import annotations

import math

from fastapi.testclient import TestClient

from tests.conftest import EPISODE_ID


def test_get_attention_success(client: TestClient) -> None:
    resp = client.get(
        f"/api/episodes/{EPISODE_ID}/timesteps/0/attention",
        params={"layer": 0, "head": 0, "action": 0},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["row"]) == 867
    assert math.isclose(sum(data["row"]), 1.0, rel_tol=1e-4)


def test_get_attention_breakdown_structure(client: TestClient) -> None:
    resp = client.get(
        f"/api/episodes/{EPISODE_ID}/timesteps/0/attention",
        params={"layer": 3, "head": 2, "action": 5},
    )
    data = resp.json()
    bd = data["breakdown"]
    assert "base_0_rgb" in bd["cameras"]
    assert len(bd["cameras"]["base_0_rgb"]) == 256
    assert len(bd["language_weights"]) == 48
    assert len(bd["action_weights"]) == 50
    assert isinstance(bd["state_weight"], float)


def test_get_attention_breakdown_totals_sum(client: TestClient) -> None:
    resp = client.get(
        f"/api/episodes/{EPISODE_ID}/timesteps/0/attention",
        params={"layer": 0, "head": 0, "action": 0},
    )
    bd = resp.json()["breakdown"]
    total = (
        sum(bd["camera_totals"].values())
        + bd["language_total"]
        + bd["state_weight"]
        + bd["action_total"]
    )
    assert math.isclose(total, 1.0, rel_tol=1e-4)


def test_get_attention_invalid_layer(client: TestClient) -> None:
    resp = client.get(
        f"/api/episodes/{EPISODE_ID}/timesteps/0/attention",
        params={"layer": 99, "head": 0, "action": 0},
    )
    assert resp.status_code == 422


def test_get_attention_invalid_head(client: TestClient) -> None:
    resp = client.get(
        f"/api/episodes/{EPISODE_ID}/timesteps/0/attention",
        params={"layer": 0, "head": 10, "action": 0},
    )
    assert resp.status_code == 422


def test_get_attention_invalid_action(client: TestClient) -> None:
    resp = client.get(
        f"/api/episodes/{EPISODE_ID}/timesteps/0/attention",
        params={"layer": 0, "head": 0, "action": 55},
    )
    assert resp.status_code == 422


def test_get_attention_missing_episode(client: TestClient) -> None:
    resp = client.get(
        "/api/episodes/nonexistent/timesteps/0/attention",
        params={"layer": 0, "head": 0, "action": 0},
    )
    assert resp.status_code == 404


def test_get_attention_summary_success(client: TestClient) -> None:
    resp = client.get(
        f"/api/episodes/{EPISODE_ID}/timesteps/0/attention/summary",
        params={"layer": 0, "head": 0},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "modality_totals" in data
    assert len(data["per_action"]) == 50
    totals = data["modality_totals"]
    assert math.isclose(sum(totals.values()), 1.0, rel_tol=1e-4)


def test_get_attention_summary_invalid_layer(client: TestClient) -> None:
    resp = client.get(
        f"/api/episodes/{EPISODE_ID}/timesteps/0/attention/summary",
        params={"layer": 42, "head": 0},
    )
    assert resp.status_code == 422


def test_get_attention_summary_invalid_head(client: TestClient) -> None:
    resp = client.get(
        f"/api/episodes/{EPISODE_ID}/timesteps/0/attention/summary",
        params={"layer": 0, "head": -1},
    )
    assert resp.status_code == 422
