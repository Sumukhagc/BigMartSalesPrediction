from src.BigMartSalesPrediction.components.data_transformation import DataTransformation
from src.BigMartSalesPrediction.config.configuration import ConfigurationManager
class DataTransformationPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config=ConfigurationManager()
        data_transformation_config=config.get_data_transformation_config()
        data_transformation=DataTransformation(data_transformation_config)
        data_transformation.preprocess()    