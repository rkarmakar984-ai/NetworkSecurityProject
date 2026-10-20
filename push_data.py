import os
import sys
import json
import certifi
import pandas as pd
import pymongo
from dotenv import load_dotenv

from networksecurity.exception.custom_exception import NetworkSecurityException
from networksecurity.logging.logger import logger

load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
ca = certifi.where()


class NetworkDataExtract:
    def csv_to_json_convertor(self, file_path):
        try:
            logger.info(f"Reading CSV file: {file_path}")
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            logger.info(f"Converted {len(records)} rows from {file_path} into records")
            return records
        except Exception as e:
            logger.error(f"Failed to convert CSV at {file_path}: {e}")
            raise NetworkSecurityException(e, sys)

    def insert_data_mongoDB(self, records, database, collection):
        try:
            if not records:
                raise ValueError("No records to insert")
            if not MONGO_DB_URL:
                raise ValueError("MONGO_DB_URL is not set (check your .env)")

            logger.info(f"Connecting to MongoDB; target {database}.{collection}")
            client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
            try:
                coll = client[database][collection]
                result = coll.insert_many(records)
                inserted = len(result.inserted_ids)
                logger.info(f"Inserted {inserted} documents into {database}.{collection}")
                return inserted
            finally:
                client.close()
                logger.info("MongoDB connection closed")
        except Exception as e:
            logger.error(f"Failed to insert data into {database}.{collection}: {e}")
            raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    FILE_PATH = os.path.join("Network_Data/phisingData.csv")
    DATABASE = "rkarmakar984-ai"
    COLLECTION = "NetworkData"

    logger.info("Starting NetworkDataExtract pipeline")
    networkobj = NetworkDataExtract()
    records = networkobj.csv_to_json_convertor(file_path=FILE_PATH)
    inserted = networkobj.insert_data_mongoDB(records, DATABASE, COLLECTION)
    print(f"Inserted {inserted} records")
    logger.info(f"Pipeline finished; inserted {inserted} records")