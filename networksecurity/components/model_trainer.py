import sys
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
)
from networksecurity.utils.main_utils.utils import save_object, load_numpy_array_data, evaluate_models
from networksecurity.exception.custom_exception import NetworkSecurityException
from networksecurity.logging.logger import logger


class ModelTrainer:
    """Trains, tunes and selects the best classifier for the pipeline."""
    
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, model_trainer_config: ModelTrainerConfig) -> None:
        """Stores the transformation artifact and model traier config"""
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
            logger.info("ModelTrainer initialized")
        
        except Exception as e:
            logger.error("Failed to initialize ModelTrainer: %s", e)
            raise NetworkSecurityException(e, sys)
    
    
    @staticmethod
    def _get_models() -> dict:
        """The candidate estimators, all seeded for reproducibility."""
        return {
            "LOGISTIC REGRESSION": LogisticRegression(max_iter=1000, random_state=42),
            "RIDGE CLASSIFIER": RidgeClassifier(),
            "SVC": SVC(probability=True, random_state=42),
            "KNN CLASSIFIER": KNeighborsClassifier(),
            "DECISION TREE CLASSIFIER": DecisionTreeClassifier(random_state=42),
            "RANDOM FOREST CLASSIFIER": RandomForestClassifier(random_state=42),
            "ADABOOST CLASSIFIER": AdaBoostClassifier(random_state=42),
            "GRADIENT BOOSTING CLASSIFIER": GradientBoostingClassifier(random_state=42),
            "CATBOOST CLASSIFIER": CatBoostClassifier(verbose=0, random_seed=42),
            "XGBOOST CLASSIFIER": XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1)
        }
    
    
    @staticmethod
    def _get_parameter_grids() -> dict:
        """Hyperparameter search spaces for each candidate model."""
        return {
            "LOGISTIC REGRESSION": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "penalty": ["l2"],
                "solver": ["lbfgs", "liblinear"]
            },
            "RIDGE CLASSIFIER": {
                "alpha": [0.01, 0.1, 1.0, 10.0],
                "solver": ["auto", "lsqr", "sag"]
            },
            "SVC": {
                "C": [0.1, 1.0, 10.0],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"]
            },
            "KNN CLASSIFIER": {
                "n_neighbors": [3, 5, 7, 11],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"]
            },
            "DECISION TREE CLASSIFIER": {
                "criterion": ["gini", "entropy"],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10]
            },
            "RANDOM FOREST CLASSIFIER": {
                "n_estimators": [8, 16, 32, 64, 128, 256],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5]
            },
            "ADABOOST CLASSIFIER": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 1.0]
            },
            "GRADIENT BOOSTING CLASSIFIER": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5]
            },
            "CATBOOST CLASSIFIER": {
                "iterations": [100, 300, 500],
                "learning_rate": [0.01, 0.1],
                "depth": [4, 6, 8]
            },
            "XGBOOST CLASSIFIER": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 1.0]
            }
        }
    
    
    def model_trainer(self, X_train, y_train, X_test, y_test) -> ModelTrainerArtifact:
        """
        Runs baseline CV, tunes the top 3 and scores the winner on the test set
        Returns a MdelTrainerArtifact Carrying the saved model path and the trained test classification metrics.
        """
        try:
            logger.info("Starting model training: %d training rows, %d test rows", len(X_train), len(X_test))
            report: dict = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=self._get_models(),
                parameters=self._get_parameter_grids(),
                top_n=3, cv_folds=5, n_iter=20,
                scoring="f1", random_state=42
            )
            
            logger.info("Baseline report: %s", report["baseline_report"])
            logger.info("Tuned report: %s", report["tuned_report"])
            
            train_metric_artifact = ClassificationMetricArtifact(**report["train_metrics"])
            test_metric_artifact = ClassificationMetricArtifact(**report["test_metrics"])
            
            if test_metric_artifact.f1_score < self.model_trainer_config.expected_accuracy:
                raise Exception(
                    f"Best model f1 {test_metric_artifact.f1_score:.4f} is below the "
                    f"expected {self.model_trainer_config.expected_accuracy}"
                )
            gap = abs(train_metric_artifact.f1_score - test_metric_artifact.f1_score)
            if gap > self.model_trainer_config.overfitting_underfitting_threshold:
                logger.info(
                    "Train/test f1 gap of %.4f exceeds %.4f, the model may be overfitting",
                    gap, self.model_trainer_config.overfitting_underfitting_threshold
                )
            
            save_object(self.model_trainer_config.trained_model_file_path, report["best_model"])
            logger.info(
                "Saved best model (%s) to: %s", report["best_model_name"],
                self.model_trainer_config.trained_model_file_path
            )
            
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric_artifact,
                test_metric_artifact=test_metric_artifact
            )
            logger.info("Model training complete: %s", model_trainer_artifact)
            return model_trainer_artifact
        
        except Exception as e:
            logger.error("Model Training failed: %s", e)
            raise NetworkSecurityException(e, sys)
    
    
    def initiate_model_training(self) -> ModelTrainerArtifact:
        """Loads the transformed arrays and runs the model trainer"""
        try:
            logger.info("Loading transformed train/test arrays")
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)
            logger.info("Train array %s | test array %s", train_arr.shape, test_arr.shape)
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            return  self.model_trainer(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        
        except Exception as e:
            logger.error("iniitiate_model_training_failed: %s", e)
            raise NetworkSecurityException(e, sys)