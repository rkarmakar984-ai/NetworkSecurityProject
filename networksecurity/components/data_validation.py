import os
import sys
import pandas as pd
from scipy.stats import ks_2samp

from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file
from networksecurity.exception.custom_exception import NetworkSecurityException
from networksecurity.logging.logger import logger


class DataValidation:
    """Validates ingested data against the schema and checks for train/test drift."""

    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """Store the ingestion artifact and validation config, and load the schema."""
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            logger.info("DataValidation initialised; schema loaded from %s", SCHEMA_FILE_PATH)
            
        except Exception as e:
            logger.error("Failed to initialise DataValidation: %s", e)
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """Read a CSV file into a pandas DataFrame."""
        try:
            logger.info("Reading data from: %s", file_path)
            df = pd.read_csv(file_path)
            logger.info("Read %d rows x %d columns from %s", df.shape[0], df.shape[1], file_path)
            return df
        
        except Exception as e:
            logger.error("Failed to read data from %s: %s", file_path, e)
            raise NetworkSecurityException(e, sys)

    def _declared_columns(self) -> dict:
        """
        Flatten the schema's `columns` into {column_name: declared_dtype}.
        Handles both a list of single-key dicts (as in this schema.yaml) and a
        plain mapping, so either layout works.
        """
        raw = self._schema_config.get("columns", {})
        if isinstance(raw, dict):
            return raw
        merged = dict()
        for item in raw:
            merged.update(item)
        return merged

    def validate_no_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """Return True if the dataframe has exactly the number of columns the schema declares."""
        try:
            expected = len(self._declared_columns())
            actual = len(dataframe.columns)
            logger.info("Required no. of columns: %d | DataFrame has: %d", expected, actual)
            return actual == expected
        
        except Exception as e:
            logger.error("Failed to validate column count: %s", e)
            raise NetworkSecurityException(e, sys)


    def validate_numerical_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Return True if every column under `numerical_columns` exists and is numeric.
        Accepts ANY numeric dtype (int64, int32, float64, ...), not just int64.
        """
        try:
            numerical_columns = self._schema_config.get("numerical_columns", [])
            logger.info("Validating %d numerical columns", len(numerical_columns))

            missing = [c for c in numerical_columns if c not in dataframe.columns]
            non_numeric = [c for c in numerical_columns
                if c in dataframe.columns and not pd.api.types.is_numeric_dtype(dataframe[c])]

            if missing:
                logger.info("Numerical columns missing from dataframe: %s", missing)
            if non_numeric:
                logger.info("Columns declared numerical but not numeric: %s", non_numeric)

            return not missing and not non_numeric
        
        except Exception as e:
            logger.error("Failed to validate numerical columns: %s", e)
            raise NetworkSecurityException(e, sys)


    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05) -> bool:
        """
        Run a KS test per numeric column and write a drift report.
        Returns True when no column drifts (every p-value >= threshold).
        Non-numeric columns are skipped, since the KS test needs numbers.
        """
        try:
            status, report = True, dict()
            for column in base_df.columns:
                if not pd.api.types.is_numeric_dtype(base_df[column]):
                    logger.info("Skipping drift test for non-numeric column '%s'", column)
                    continue

                df1, df2 = base_df[column], current_df[column]
                is_same_dist = ks_2samp(df1, df2)
                p_value = float(is_same_dist.pvalue)
                if threshold <= p_value:
                    is_found = False
                else:
                    is_found = True
                    status = False
                    logger.info("Drift detected in column '%s' (p=%.4f < %.2f)", column, p_value, threshold)
                report.update({column: {"p_value": p_value, "drift_status": is_found}})

            drift_report_file_path = self.data_validation_config.drift_report_file_path
            write_yaml_file(file_path=drift_report_file_path, content=report)
            logger.info("Drift report written to: %s | overall drift status=%s", drift_report_file_path, status)
            return status
        
        except Exception as e:
            logger.error("Failed to detect dataset drift: %s", e)
            raise NetworkSecurityException(e, sys)


    def initiate_data_validation(self) -> DataValidationArtifact:
        """Run the full validation flow and return the resulting DataValidationArtifact."""
        try:
            logger.info("Starting data validation")
            train_dataframe = DataValidation.read_data(self.data_ingestion_artifact.train_file_path)
            test_dataframe = DataValidation.read_data(self.data_ingestion_artifact.test_file_path)

            train_status = self.validate_no_of_columns(dataframe=train_dataframe)
            test_status = self.validate_no_of_columns(dataframe=test_dataframe)

            train_numerical_status = self.validate_numerical_columns(dataframe=train_dataframe)
            test_numerical_status = self.validate_numerical_columns(dataframe=test_dataframe)

            if not train_status:
                logger.info("Train dataframe does not contain all expected columns")
            if not test_status:
                logger.info("Test dataframe does not contain all expected columns")

            if train_status and test_status:
                drift_status = self.detect_dataset_drift(base_df=train_dataframe, current_df=test_dataframe)
            else:
                logger.info("Skipping drift detection because column validation failed")
                drift_status = False

            validation_status = bool(
                train_status and test_status and train_numerical_status
                and test_numerical_status and drift_status
            )

            if validation_status:
                logger.info("Validation passed; saving data to the validated directory")
                train_out = self.data_validation_config.valid_train_file_path
                test_out = self.data_validation_config.valid_test_file_path
                valid_train_out, valid_test_out = train_out, test_out
                invalid_train_out = invalid_test_out = None
            else:
                logger.info("Validation failed; saving data to the invalid directory")
                train_out = self.data_validation_config.invalid_train_file_path
                test_out = self.data_validation_config.invalid_test_file_path
                valid_train_out = valid_test_out = None
                invalid_train_out, invalid_test_out = train_out, test_out

            os.makedirs(os.path.dirname(train_out), exist_ok=True)
            train_dataframe.to_csv(train_out, index=False, header=True)
            logger.info("Saved train data to: %s", train_out)

            os.makedirs(os.path.dirname(test_out), exist_ok=True)
            test_dataframe.to_csv(test_out, index=False, header=True)
            logger.info("Saved test data to: %s", test_out)

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=valid_train_out,
                valid_test_file_path=valid_test_out,
                invalid_train_file_path=invalid_train_out,
                invalid_test_file_path=invalid_test_out,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            logger.info("Data validation complete: %s", data_validation_artifact)
            return data_validation_artifact

        except Exception as e:
            logger.error("Data validation failed: %s", e)
            raise NetworkSecurityException(e, sys)