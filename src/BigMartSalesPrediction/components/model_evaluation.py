from src.BigMartSalesPrediction.entity import ModelEvaluationConfig
from src.BigMartSalesPrediction.utils.common import load_model
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
class ModelEvaluation:
    def __init__(self,config:ModelEvaluationConfig) -> None:
        self.config=config
    def save_score(self,metrics:dict):
        with open(self.config.evaluation_path,'a') as f:
            for metric in metrics.keys():
                f.write(metric+"\t")
                f.write(str(metrics[metric])+"\n")

                

         

    def evaluate(self):
        x_test=load_model(self.config.x_test_path)
        y_test=load_model(self.config.y_test_path)
        model=load_model(self.config.model_path)
        y_pred=model.predict(x_test)
        metrics={}
        metrics['r2score']=r2_score(y_test,y_pred)
        metrics['mean_squared_error']=mean_squared_error(y_test,y_pred)
        metrics['mean_absolute_error']=mean_absolute_error(y_test,y_pred)
        self.save_score(metrics)