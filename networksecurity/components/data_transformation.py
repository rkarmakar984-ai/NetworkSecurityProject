import sys
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from networksecurity.constant.training_pipeline import (
    TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMETERS
)
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.entity.artifact_entity import (
    DataValidationArtifact, DataTransformationArtifact
)
from networksecurity.utils.main_utils.utils import (
    save_numpy_array_data, save_object
)
from networksecurity.exception.custom_exception import NetworkSecurityException
from networksecurity.logging.logger import logger


class DataTransformation:
    """Transforms validated data into model_ready NumPy arrays"""
    
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig) -> None:
        """Stores the validation artifact and transformation configuration"""
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            logger.info("DataTransformation initialised")
        
        except Exception as e:
            logger.error("Failed to initialise DataTransformation: %s", e)
            raise NetworkSecurityException(e, sys)


    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """Reads a CSV file into a pandas DataFrame"""
        try:
            logger.info("Reading data from: %s", file_path)
            df = pd.read_csv(file_path)
            logger.info("Read %d rows x %d columns from %s", df.shape[0], df.shape[1], file_path)
            return df
        
        except Exception as e:
            logger.error("Failed to read data from %s: %s", file_path, e)
            raise NetworkSecurityException(e, sys)


    @staticmethod
    def imputer_pipeline() -> Pipeline:
        """
        Build the preprocessing pipeline (a single KNNImputer step)
        Imputer settings come from DATA_TRANSFORMATION_IMPUTER_PARAMETERS from the constants file
        """
        try:
            logger.info("Building preprocessor with KNNImputer paramaters: %s", DATA_TRANSFORMATION_IMPUTER_PARAMETERS)
            imputer: KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMETERS)
            processor: Pipeline = Pipeline([("IMPUTER", imputer)])
            logger.info("Preprocessor pipeline created")
            return processor
        
        except Exception as e:
            logger.error("Falied to build the data transforming KNNImputer pipeline: %s", e)
            raise NetworkSecurityException(e, sys)


    def initiate_data_transformation(self, ) -> DataTransformationArtifact:
        """Runs the full data transformation flow and returns the resulting DataTransformationArtifact"""
        try:
            logger.info("Starting data transformation")
            if not self.data_validation_artifact.validation_status:
                raise Exception("Data Validation didn't pass, refusing to transform invalid data")
            
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
            target_feature_train_df = train_df[TARGET_COLUMN].replace(-1, 0)
            logger.info(
                "Train: %d features, target counts %s",
                input_feature_train_df.shape[1],
                target_feature_train_df.value_counts().to_dict()
            )
            
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])
            target_feature_test_df = test_df[TARGET_COLUMN].replace(-1, 0)
            logger.info(
                "Test: %d features, target counts %s",
                input_feature_test_df.shape[1],
                target_feature_test_df.value_counts().to_dict()
            )
            
            preprocessor = DataTransformation.imputer_pipeline()
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            logger.info("Preprocessor fitted on training features")
            
            transfromed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)
            logger.info(
                "Transformed features -> train %s, test %s",
                transfromed_input_train_feature.shape,
                transformed_input_test_feature.shape
            )
            
            train_arr = np.c_[transfromed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]
            logger.info("Final arrays -> train %s, test %s", train_arr.shape, test_arr.shape)
            
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)
            save_object("final_model/preprocessor.pkl", preprocessor_object)
            logger.info("Saved transformed arrays and preprocessor object")
            
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            logger.info("Data transformation complete: %s", data_transformation_artifact)
            return data_transformation_artifact       
        
        except Exception as e:
            logger.error("Data Transformation failed: %s", e)
            raise NetworkSecurityException(e, sys)