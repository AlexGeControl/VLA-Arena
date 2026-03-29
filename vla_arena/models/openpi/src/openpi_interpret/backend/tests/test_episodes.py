"""Tests for the episodes router (C3): listing, detail, camera, token-meta."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import EPISODE_ID, INSTRUCTION_TOKENS, NUM_TIMESTEPS, TASK_INSTRUCTION


def test_list_episodes_returns_all(client: TestClient) -> None:
    resp = client.get("/api/episodes")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["episode_id"] == EPISODE_ID
    assert data[0]["task_instruction"] == TASK_INSTRUCTION
    assert data[0]["num_timesteps"] == NUM_TIMESTEPS


def test_get_episode_detail_success(client: TestClient) -> None:
    resp = client.get(f"/api/episodes/{EPISODE_ID}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["episode_id"] == EPISODE_ID
    assert data["instruction_tokens"] == INSTRUCTION_TOKENS
    assert data["sampled_layers"] == [0, 3, 6, 9, 12, 15, 17]
    assert isinstance(data["camera_names"], list)


def test_get_episode_detail_not_found(client: TestClient) -> None:
    resp = client.get("/api/episodes/nonexistent")
    assert resp.status_code == 404


def test_get_camera_image_success(client: TestClient) -> None:
    resp = client.get(f"/api/episodes/{EPISODE_ID}/camera/base_0_rgb")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert len(resp.content) > 0


def test_get_camera_image_missing_camera(client: TestClient) -> None:
    resp = client.get(f"/api/episodes/{EPISODE_ID}/camera/nonexistent_cam")
    assert resp.status_code == 404


def test_get_camera_image_missing_episode(client: TestClient) -> None:
    resp = client.get("/api/episodes/bad_id/camera/base_0_rgb")
    assert resp.status_code == 404


def test_get_token_meta_success(client: TestClient) -> None:
    resp = client.get(
        f"/api/episodes/{EPISODE_ID}/timesteps/0/token-meta"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 867
    assert data[0]["type"] == "image_patch"
    assert data[0]["source"] == "base_0_rgb"


def test_get_token_meta_language_tokens(client: TestClient) -> None:
    resp = client.get(
        f"/api/episodes/{EPISODE_ID}/timesteps/0/token-meta"
    )
    data = resp.json()
    lang_tokens = [t for t in data if t["type"] == "language"]
    assert len(lang_tokens) == 48
    assert lang_tokens[0]["token_text"] == "pick"


def test_get_token_meta_missing_timestep(client: TestClient) -> None:
    resp = client.get(
        f"/api/episodes/{EPISODE_ID}/timesteps/99/token-meta"
    )
    assert resp.status_code == 404


def test_get_token_meta_missing_episode(client: TestClient) -> None:
    resp = client.get(
        "/api/episodes/nonexistent/timesteps/0/token-meta"
    )
    assert resp.status_code == 404


def test_health_endpoint(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
