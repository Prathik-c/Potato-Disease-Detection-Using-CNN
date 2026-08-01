"""
Tests for the plant disease detection API.
Run with: backend/venv/Scripts/python -m pytest backend/tests/ -v
"""
import io
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np


@pytest.fixture
def client():
    """Create a test client with a mocked model service."""
    with patch("app.services.model_service.model_service.load"):
        from app.main import app
        return TestClient(app)


def test_health_check(client):
    """Health endpoint should return 200 with status info."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_predict_no_file(client):
    """Predict endpoint should return 422 when no file is provided."""
    response = client.post("/predict")
    assert response.status_code == 422


def test_predict_invalid_file(client):
    """Predict endpoint should return 400 for non-image files."""
    fake_file = io.BytesIO(b"this is not an image")
    response = client.post(
        "/predict",
        files={"file": ("test.txt", fake_file, "text/plain")}
    )
    assert response.status_code == 400


def test_predict_returns_expected_schema(client):
    """Predict endpoint should return label and confidence keys."""
    import cv2

    # Create a small green test image (will trigger leaf detection)
    green_img = np.zeros((100, 100, 3), dtype=np.uint8)
    green_img[:, :] = [50, 150, 50]  # BGR green
    _, encoded = cv2.imencode(".jpg", green_img)
    img_bytes = encoded.tobytes()

    with patch("app.api.endpoints.model_service") as mock_model:
        mock_model.predict.return_value = ("healthy", 0.95)
        response = client.post(
            "/predict",
            files={"file": ("leaf.jpg", io.BytesIO(img_bytes), "image/jpeg")}
        )

    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "confidence" in data
