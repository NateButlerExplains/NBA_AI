"""
test_pipeline.py

Smoke tests for the production pipeline components.
Verifies that all pipeline modules can be imported and that
key integration points (predictor registry, web app) work.
"""

import pytest


class TestPipelineImports:
    """Verify that all pipeline modules can be imported."""

    def test_phase5_predictor_imports(self):
        from src.pipeline.phase5_predictor import Phase5Predictor

    def test_phase3_predictor_imports(self):
        from src.pipeline.phase3_predictor import Phase3Predictor

    def test_ensemble_predictor_imports(self):
        from src.pipeline.ensemble_predictor import EnsemblePredictor

    def test_orchestrator_imports(self):
        from src.pipeline.orchestrator import PipelineOrchestrator


class TestPredictorRegistry:
    """Verify that all predictors are registered in the prediction manager."""

    def test_all_predictors_registered(self):
        from src.predictions.prediction_manager import VALID_PREDICTORS

        expected = {"Baseline", "Linear", "Tree", "MLP", "Phase5", "Phase3", "Ensemble"}
        assert expected == VALID_PREDICTORS


class TestWebAppIntegration:
    """Verify that the web app creates and serves basic routes."""

    def test_web_app_creates(self):
        from src.web_app.app import create_app

        app = create_app("Phase5")
        assert app is not None

    def test_dashboard_endpoint(self):
        from src.web_app.app import create_app

        app = create_app("Phase5")
        app.config["TESTING"] = True
        with app.test_client() as client:
            r = client.get("/dashboard?predictor=Phase5")
            assert r.status_code == 200
