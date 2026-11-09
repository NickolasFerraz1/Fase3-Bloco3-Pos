"""Teste end-to-end do pipeline."""
from pathlib import Path
from pipeline import run_pipeline


def test_pipeline_executa_completo(tmp_path, monkeypatch):
    """Pipeline roda do início ao fim e gera modelo."""
    monkeypatch.chdir(tmp_path)

    metrics = run_pipeline()

    # Modelo foi salvo
    assert Path("models/iris_model.pkl").exists()

    # Accuracy aceitável
    assert metrics["accuracy"] >= 0.85
