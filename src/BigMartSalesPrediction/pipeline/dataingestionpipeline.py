from src.BigMartSalesPrediction.config.configuration import ConfigurationManager
from src.BigMartSalesPrediction.components.data_ingestion import DataIngestion
class DataIngestionPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config=ConfigurationManager()
        data_ingestion_config=config.get_data_ingestion_config()
        data_ingestion=DataIngestion(data_ingestion_config)
        data_ingestion.download_data()
        data_ingestion.unzip_data()
