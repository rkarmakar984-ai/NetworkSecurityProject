from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionArtifact:
    """Output of Data Ingestion: paths to the train/test split CSVs."""
    train_file_path: str
    test_file_path: str


@dataclass(frozen=True)
class DataValidationArtifact:
    """Output of Data Validation: overall status, the valid/invalid split paths, and the drift report path."""
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str


@dataclass(frozen=True)
class DataTransformationArtifact:
    """Output of Data Transformation: the fitted preprocessor object and the transformed splits."""
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str


@dataclass(frozen=True)
class ClassificationMetricArtifact:
    """Classification metrics for a single dataset (train or test)."""
    f1_score: float
    precision_score: float
    recall_score: float


@dataclass(frozen=True)
class ModelTrainerArtifact:
    """Output of Model Trainer: the saved model file and its train/test metric artifacts."""
    trained_model_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact