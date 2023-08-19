from src.BigMartSalesPrediction.utils.common import read_yaml,create_directories,getsize
from src.BigMartSalesPrediction.constants import *
from src.BigMartSalesPrediction.entity import DataIngestionConfig,DataTransformationConfig,ModelTrainerConfig,ModelEvaluationConfig
class ConfigurationManager:
    def __init__(self,config_file_path=CONFIG_FILE_PATH,params_file_path=PARAMS_FILE_PATH) -> DataIngestionConfig:
        self.config=read_yaml(config_file_path)
        self.params=read_yaml(params_file_path)
        create_directories([self.config['artifacts_root']])

    def get_data_ingestion_config(self) ->DataIngestionConfig:
        config=self.config['data_ingestion']
        create_directories([config['root_dir']])
        return DataIngestionConfig(config['root_dir'],config['source_url'],config['local_dir'],config['local_dir'])

    def get_data_transformation_config(self)->DataTransformationConfig:
        config=self.config['data_transformation']
        create_directories([config['root_dir']])
        return DataTransformationConfig(config['root_dir'],config['data_path'],config['x_train_path'],config['x_test_path'],config['y_train_path'],config['y_test_path'])
    
    def get_model_trainer_config(self)->ModelTrainerConfig:
        config=self.config['model_trainer']
        create_directories([config['root_dir']])
        return ModelTrainerConfig(config['root_dir'],config['model_path'],config['x_train_path'],config['y_train_path'])
    
    def get_model_evaluation_config(self)->ModelEvaluationConfig:
        config=self.config['model_evaluation']
        create_directories([config['root_dir']])
        return ModelEvaluationConfig(config['root_dir'],config['model_path'],config['x_test_path'],config['y_test_path'],config['evaluation_path'])
    