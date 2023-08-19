from src.BigMartSalesPrediction.components.model_trainer import ModelTrainer
from src.BigMartSalesPrediction.config.configuration import ConfigurationManager

class ModelTrainerPipeline:
    def main(self):
        config=ConfigurationManager()
        model_trainer_config=config.get_model_trainer_config()
        model_trainer=ModelTrainer(model_trainer_config)
        model_trainer.train_model()
