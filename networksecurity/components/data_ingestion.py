import os
import sys
import numpy as np
import pandas as pd
import pymongo
import certifi
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact
from networksecurity.exception.custom_exception import NetworkSecurityException
from networksecurity.logging.logger import logger

load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class DataIngestion:
    """Reads data from MongoDB, saves it as a feature store CSV, and produces the train/test split."""
    
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """Store the ingestion configuration for the pipeline methods to use."""
        try:
            self.data_ingestion_config = data_ingestion_config
            logger.info("DataIngestion initialised with config: %s", self.data_ingestion_config)
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """
        Reads the configured MongoDB collection into a pandas DataFrame.
        """
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name

            if not MONGO_DB_URL:
                raise ValueError("MONGO_DB_URL is not set (check your .env)")

            logger.info("Reading from MongoDB collection %s.%s", database_name, collection_name)
            
            # Using certifi for safe SSL connection verification
            ca = certifi.where()
            client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
            try:
                collection = client[database_name][collection_name]
                df = pd.DataFrame(list(collection.find()))
                if df.empty:
                    raise ValueError(f"Collection {database_name}.{collection_name} is empty.")
            finally:
                client.close()

            logger.info("Read %d rows x %d columns from %s.%s",
                        df.shape[0], df.shape[1], database_name, collection_name)

            if "_id" in df.columns:
                df = df.drop(columns=["_id"], axis=1)
                logger.info("Dropped the MongoDB '_id' column")

            df.replace({"na": np.nan}, inplace=True)
            
            return df
        
        except Exception as e:
            logger.error("Failed to read collection as dataframe: %s", e)
            raise NetworkSecurityException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Persist the raw dataframe to the feature store CSV and return it unchanged."""
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logger.info("Saving feature store file to: %s", feature_store_file_path)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            logger.info("Feature store file saved (%d rows)", len(dataframe))
            
            return dataframe
        
        except Exception as e:
            logger.error("Failed to export data into feature store: %s", e)
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        """Split the dataframe into train/test sets and write each to its configured CSV path."""
        try:
            logger.info("Splitting dataframe with test_size=%s",
                        self.data_ingestion_config.train_test_split_ratio)
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42,
            )
            logger.info("Split complete: train=%d rows, test=%d rows", len(train_set), len(test_set))

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logger.info("Exporting train and test file paths")
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logger.info("Exported train and test file paths")
            
        except Exception as e:
            logger.error("Failed to split data into train and test: %s", e)
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """Run the full ingestion flow and return the resulting DataIngestionArtifact."""
        try:
            logger.info("Starting data ingestion")
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
            logger.info("Data ingestion complete: %s", data_ingestion_artifact)
            
            return data_ingestion_artifact
        
        except Exception as e:
            logger.error("Data ingestion failed: %s", e)
            raise NetworkSecurityException(e, sys)