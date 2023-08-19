from src.BigMartSalesPrediction.components.model_evaluation import ModelEvaluation
from src.BigMartSalesPrediction.config.configuration import ConfigurationManager

class ModelEvaluationPipeline:
    def main(self):
        config=ConfigurationManager()
        model_evaluation_config=config.get_model_evaluation_config()
        model_evaluation=ModelEvaluation(model_evaluation_config)
        model_evaluation.evaluate()