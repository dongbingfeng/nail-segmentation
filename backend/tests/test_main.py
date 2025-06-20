import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from server.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_crud():
    # Create
    item = {"name": "test", "description": "desc", "value": 1.0}
    resp = client.post("/items", json=item)
    assert resp.status_code == 201
    data = resp.json()
    item_id = data["id"]
    # Read
    resp = client.get(f"/items/{item_id}")
    assert resp.status_code == 200
    # Update
    item["value"] = 2.0
    resp = client.put(f"/items/{item_id}", json=item)
    assert resp.status_code == 200
    # Delete
    resp = client.delete(f"/items/{item_id}")
    assert resp.status_code == 200
    # Not found after delete
    resp = client.get(f"/items/{item_id}")
    assert resp.status_code == 404 