"""Tests for the embedding router (C5): t-SNE points and neighbors."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.data.constants import MODALITY_GROUPS, TOKEN_COLORS, TOKEN_RANGES
from tests.conftest import EPISODE_ID


def test_get_tsne_success(client: TestClient) -> None:
    resp = client.get(
        f"/api/episodes/{EPISODE_ID}/timesteps/0/tsne",
        params={"layer": 0},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["points"]) == 867


def test_get_tsne_point_structure(client: TestClient) -> None:
    resp = client.get(
        f"/api/episodes/{EPISODE_ID}/timesteps/0/tsne",
        params={"layer": 0},
    )
    pt = resp.json()["points"][0]
    assert "index" in pt
    assert "x" in pt
    assert "y" in pt
    assert "type" in pt
    assert "source" in pt
    assert "color" in pt


def test_get_tsne_colors_match_constants(client: TestClient) -> None:
    resp = client.get(
        f"/api/episodes/{EPISODE_ID}/timesteps/0/tsne",
        params={"layer": 0},
    )
    points = resp.json()["points"]
    first_img = points[0]
    assert first_img["color"] == TOKEN_COLORS["base_0_rgb"]
    lang_point = points[TOKEN_RANGES["language"][0]]
    assert lang_point["color"] == TOKEN_COLORS["language"]


def test_get_tsne_invalid_layer(client: TestClient) -> None:
    resp = client.get(
        f"/api/episodes/{EPISODE_ID}/timesteps/0/tsne",
        params={"layer": 99},
    )
    assert resp.status_code == 422


def test_get_tsne_missing_episode(client: TestClient) -> None:
    resp = client.get(
        "/api/episodes/nonexistent/timesteps/0/tsne",
        params={"layer": 0},
    )
    assert resp.status_code == 404


def test_get_neighbors_success(client: TestClient) -> None:
    resp = client.get(
        f"/api/episodes/{EPISODE_ID}/timesteps/0/tsne/neighbors",
        params={"layer": 0, "action": 0},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "selected" in data
    assert len(data["neighbors"]) == 5


def test_get_neighbors_selected_point(client: TestClient) -> None:
    resp = client.get(
        f"/api/episodes/{EPISODE_ID}/timesteps/0/tsne/neighbors",
        params={"layer": 0, "action": 0},
    )
    selected = resp.json()["selected"]
    action_global_idx = TOKEN_RANGES["action"][0] + 0
    assert selected["index"] == action_global_idx


def test_get_neighbors_modality_groups(client: TestClient) -> None:
    resp = client.get(
        f"/api/episodes/{EPISODE_ID}/timesteps/0/tsne/neighbors",
        params={"layer": 0, "action": 0},
    )
    groups = [n["modality_group"] for n in resp.json()["neighbors"]]
    assert groups == MODALITY_GROUPS


def test_get_neighbors_invalid_layer(client: TestClient) -> None:
    resp = client.get(
        f"/api/episodes/{EPISODE_ID}/timesteps/0/tsne/neighbors",
        params={"layer": 99, "action": 0},
    )
    assert resp.status_code == 422


def test_get_neighbors_invalid_action(client: TestClient) -> None:
    resp = client.get(
        f"/api/episodes/{EPISODE_ID}/timesteps/0/tsne/neighbors",
        params={"layer": 0, "action": 55},
    )
    assert resp.status_code == 422


def test_get_neighbors_missing_episode(client: TestClient) -> None:
    resp = client.get(
        "/api/episodes/nonexistent/timesteps/0/tsne/neighbors",
        params={"layer": 0, "action": 0},
    )
    assert resp.status_code == 404


def test_get_neighbors_distance_positive(client: TestClient) -> None:
    resp = client.get(
        f"/api/episodes/{EPISODE_ID}/timesteps/0/tsne/neighbors",
        params={"layer": 0, "action": 3},
    )
    for n in resp.json()["neighbors"]:
        assert n["distance"] > 0
