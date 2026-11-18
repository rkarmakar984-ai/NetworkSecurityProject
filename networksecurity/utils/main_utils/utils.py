import os
import sys
import pickle
import yaml
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score

from networksecurity.exception.custom_exception import NetworkSecurityException
from networksecurity.logging.logger import logger

def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and return its parsed contents
    """
    try:
        logger.info("Reading YAML file: %s", file_path)
        with open(file_path, "rb") as yaml_file:
            content = yaml.safe_load(yaml_file)
        logger.info("Successfully read YAML file: %s", file_path)
        return content
    
    except Exception as e:
        logger.error("Failed to read YAML file %s: %s", file_path, e)
        raise NetworkSecurityException(e, sys)


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Writes 'content' to a YAML file, creating parent directories as needed.
    If 'replace' is True, an existing file at 'file_path' is removed first.
    """
    try:
        if replace and os.path.exists(file_path):
            logger.info("Replacing existing YAML file: %s", file_path)
            os.remove(file_path)

        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        logger.info("Writing YAML file: %s", file_path)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
        logger.info("Successfully wrote YAML file: %s", file_path)

    except Exception as e:
        logger.error("Failed to write YAML file %s: %s", file_path, e)
        raise NetworkSecurityException(e, sys)


def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    """
    Saves NumPy array data to 'file_path' (as .npy), creating parent directories
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        logger.info("Saving numpy array to: %s", file_path)
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
        logger.info("Saved numpy array (shape=%s) to: %s", array.shape, file_path)
            
    except Exception as e:
        logger.error("Failed to save numpy array to %s: %s", file_path, e)
        raise NetworkSecurityException(e, sys)


def save_object(file_path: str, obj: object) -> None:
    """Serialize 'obj' to 'file_path' using pickle, creating parent directories"""
    try:
        logger.info("Saving object to: %s", file_path)
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logger.info("Saved object to: %s", file_path)
        
    except Exception as e:
        logger.error("Failed to save object to %s: %s", file_path, e)
        raise NetworkSecurityException(e, sys)


def load_object(file_path: str) -> object:
    """Load and return the pickle-serialized object stored at 'file_path'"""
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist")
        logger.info("Loading object from: %s", file_path)
        with open(file_path, "rb") as file_obj:
            obj =  pickle.load(file_obj)
        logger.info("Loaded object from: %s", file_path)
        return obj
    
    except Exception as e:
        logger.error("Failed to load object from %s: %s", file_path, e)
        raise NetworkSecurityException(e, sys)


def load_numpy_array_data(file_path: str) -> np.ndarray:
    """Loads and returns the NumPy array stored at 'file_path'"""
    try:
        logger.info("Loading numpy array from: %s", file_path)
        with open(file_path, "rb") as file_obj:
            array = np.load(file_obj, allow_pickle=True)
        logger.info("Loaded numpy array (shape=%s) from: %s", array.shape, file_path)
        return array
    
    except Exception as e:
        logger.error("Failed to load numpy array from %s: %s", file_path, e)
        raise NetworkSecurityException(e, sys)


def _param_space_size(param_distributions: dict) -> int:
    """Returns total no. of distinct combinations in a parameter grid."""
    size = 1
    for values in param_distributions.values():
        size *= len(values)
    return size


def evaluate_models(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    models: dict, params: dict,
    top_n: int = 3, cv_folds: int = 5, n_iter: int = 20,
    scoring: str = "f1", random_state: int = 42
    ) -> dict:
    try:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        # Stage 1: baseline CV, no tuning
        logger.info("Stage 1: baseline CV of %d models (folds=%d, scoring=%s)", len(models), cv_folds, scoring)
        baseline_report = dict()
        for name, model in models.items():
            scores = cross_val_score(
                estimator=clone(model),
                X=X_train, y=y_train,
                scoring=scoring,
                cv=cv, n_jobs=-1
            )
            baseline_report[name] = float(np.mean(scores))
            logger.info(
                "  %-30s baseline %s = %.4f (+/- %.4f)",
                name, scoring, np.mean(scores), np.std(scores)
            )
        ranked = sorted(baseline_report.items(), key=lambda x: x[1], reverse=True)
        top_models = [name for name, _ in ranked[:top_n]]
        logger.info("Top %d by baseline %s: %s", len(top_models), scoring, top_models)

        # Stage 2: Tune only the leaders
        logger.info("Stage 2: RandomizedSearchCV on the top %d models", len(top_models))
        tuned_report = dict()
        best_name, best_score, best_estimator = None, -np.inf, None
        for name in top_models:
            grid = params.get(name, {})
            if grid:
                search = RandomizedSearchCV(
                    estimator=clone(model[name]),
                    param_distributions=grid,
                    n_iter=min(n_iter, _param_space_size(grid)),
                    cv=cv, random_state=random_state,
                    scoring=scoring, n_jobs=-1, refit=True
                )
                search.fit(X=X_train, y=y_train)
                estimator, score = search.best_estimator_, float(search.best_score_)
                tuned_report[name] = {
                    "best_score": score,
                    "best_parameters": search.best_params_
                }
                logger.info(
                    "  %-30s tuned %s = %.4f | best parameters: %s",
                    name, scoring, score, search.best_score_
                )
            else:
                estimator = clone(models[name]).fit(X=X_train, y=y_train)
                score = baseline_report[name]
                tuned_report[name] = {
                    "best_score": score,
                    "best_params": {}
                }
                logger.info("  %-30s no grid supplied; keeping baseline %.4f", name, score)
            if score > best_score:
                best_name, best_estimator, best_score = name, score, estimator
        logger.info("Best tuned model: %s (CV %s = %.4f)", best_name, scoring, best_score)

        # Stage 3: the test set results
        def _metrics(y_true, y_pred) -> dict:
            return {
                "f1_score": float(f1_score(y_true, y_pred, average="weighted")),
                "precision_score": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
                "recall_score": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            }
        train_metrics = _metrics(y_train, best_estimator.predict(X_train))
        test_metrics = _metrics(y_test, best_estimator.predict(X_test))
        logger.info("Train metrics: %s", train_metrics)
        logger.info("Test  metrics: %s", test_metrics)
        
        return {
            "baseline_report": baseline_report,
            "top_models": top_models,
            "tuned_report": tuned_report,
            "best_model_name": best_name,
            "best_model": best_estimator,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics
        }
    
    except Exception as e:
        logger.error("Model evaluation failed: %s", e)
        raise NetworkSecurityException(e, sys)