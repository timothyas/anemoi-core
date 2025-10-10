import pytest

from pathlib import Path
from mlflow.tracking import MlflowClient
from anemoi.training.diagnostics.mlflow.azureml import AnemoiAzureMLflowLogger, AzureIdentity


@pytest.fixture(scope="session")
def tmp_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    # returns a session-scoped temporary directory
    return str(tmp_path_factory.mktemp("mlruns"))


@pytest.fixture
def tmp_uri(monkeypatch, tmp_path):
    uri = (Path(tmp_path) / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    return uri


# @pytest.fixture
# def tmp_client(tmp_uri):
#     return MlflowClient(tracking_uri=tmp_uri)


@pytest.fixture
def default_logger(tmp_path, tmp_uri) -> AnemoiAzureMLflowLogger:
    print
    logger = AnemoiAzureMLflowLogger(
        identity=AzureIdentity("managed"),
        experiment_name="test_experiment",
        run_name="test_run",
        offline=False,
        tracking_uri=tmp_uri,
        authentication=False,
        save_dir=tmp_path,
    )
    return logger


def test_mlflowlogger_params_limit(default_logger):
    default_logger._max_params_length = 3
    params = {"lr": 0.001, "path": "era5", "anemoi.version": 1.5, "bounding": True}
    # # Expect an exception when logging too many hyperparameters
    with pytest.raises(ValueError, match=r"Too many params:"):
        default_logger.log_hyperparams(params)


def test_mlflowlogger_metric_deduplication(default_logger):
    default_logger.log_metrics({"foo": 1.0}, step=5)
    default_logger.log_metrics({"foo": 1.0}, step=5)  # duplicate
    # Only the first metric should be logged
    assert len(default_logger._logged_metrics) == 1
    assert next(iter(default_logger._logged_metrics))[0] == "foo"  # key
    assert next(iter(default_logger._logged_metrics))[1] == 5  # step

