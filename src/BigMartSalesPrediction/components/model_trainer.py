from src.BigMartSalesPrediction.logging import logger
from src.BigMartSalesPrediction.utils.common import load_model,save_obj
from src.BigMartSalesPrediction.entity import ModelTrainerConfig 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
class ModelTrainer:
    def __init__(self,config:ModelTrainerConfig) -> None:
        self.config=config

    def load_data(self):
        x_train=load_model(self.config.x_train_path)
        y_train=load_model(self.config.y_train_path)
        
        return x_train,y_train
    def find_hyperparameters(self,model:RandomForestRegressor,x_train,y_train):
        params={
            'n_estimators':[100,200,300,400,500],
            'max_features' : ['auto', 'sqrt'],
            'max_samples' :[int(x) for x in range(500,2001,500)],
            'min_samples_split' : [2, 5, 10],
            'min_samples_leaf' : [1, 2, 4]
        }
        tuned_model = RandomizedSearchCV(estimator = model, param_distributions = params, n_iter = 100,
         cv = 3, random_state=42)
        tuned_model.fit(x_train,y_train)
        return tuned_model.best_params_

    def train_model(self):
        x_train,y_train=self.load_data()
        model=RandomForestRegressor()
        params=self.find_hyperparameters(model,x_train,y_train)    
        logger.info("Finding Hyperparameters done.")
        n_estimators=params['n_estimators']
        max_features=params['max_features']
        max_samples=params['max_samples']
        min_samples_split=params['min_samples_split']
        min_samples_leaf=params['min_samples_leaf']
        model=RandomForestRegressor(n_estimators=n_estimators,max_features=max_features,max_samples=max_samples,
        min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)
        model.fit(x_train,y_train)
        save_obj(self.config.model_path,model)
        logger.info("Model training done.")
