from fastapi.testclient import TestClient

from grid_ops_backend.api.app import create_app


def test_run_flow_and_compare() -> None:
    client = TestClient(create_app())

    create_response = client.post(
        "/api/v1/runs",
        json={"network_id": "ieee14", "seed": 7},
    )
    assert create_response.status_code == 200
    run_id = create_response.json()["data"]["run_id"]

    screen_response = client.post(
        f"/api/v1/runs/{run_id}/screen",
        json={"top_k": 3},
    )
    assert screen_response.status_code == 200
    assert screen_response.json()["data"]["dangerous_count"] >= 0

    recommend_response = client.post(
        f"/api/v1/runs/{run_id}/recommend",
        json={"mode": "baseline"},
    )
    assert recommend_response.status_code == 200
    assert recommend_response.json()["data"]["accepted"] is True

    compare_response = client.post(f"/api/v1/runs/{run_id}/compare")
    assert compare_response.status_code == 200
    assert compare_response.json()["data"]["winner"] in {"baseline", "llm_assisted"}


def test_health_endpoint() -> None:
    client = TestClient(create_app())
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert "pandapower_available" in body["data"]
