from src.BigMartSalesPrediction.config.configuration import DataTransformationConfig
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.BigMartSalesPrediction.utils.common import save_obj
from src.BigMartSalesPrediction.logging import logger
class DataTransformation:
    def __init__(self,config:DataTransformationConfig) -> None:
        self.config=config

    def load_data(self):
        data=pd.read_csv(self.config.data_path)
        print(type(data))
        return data
    def treat_null_values(self,data:pd.DataFrame):
        mean_item_weight=data['Item_Weight'].mean()
        data['Item_Weight'].fillna(mean_item_weight,inplace=True)
        mode_outlet_size=data['Outlet_Size'].mode()[0]
        data['Outlet_Size'].fillna(mode_outlet_size,inplace=True)

    def treat_categorical_data(self,data:pd.DataFrame):
        fat_content=data['Item_Fat_Content'].value_counts().to_dict()
        item_type=data['Item_Type'].value_counts().to_dict()
        outlet_identifier=data['Outlet_Identifier'].value_counts().to_dict()
        outlet_size=data['Outlet_Size'].value_counts().to_dict()
        outlet_location_type=data['Outlet_Location_Type'].value_counts().to_dict()
        outlet_type=data['Outlet_Type'].value_counts().to_dict()
        data['Item_Fat_Content']=data['Item_Fat_Content'].map(fat_content)
        data['Item_Type']=data['Item_Type'].map(item_type)
        data['Outlet_Identifier']=data['Outlet_Identifier'].map(outlet_identifier)  
        data['Outlet_Size']=data['Outlet_Size'].map(outlet_size)
        data['Outlet_Location_Type']=data['Outlet_Location_Type'].map(outlet_location_type)
        data['Outlet_Type']=data['Outlet_Type'].map(outlet_type)
        return data
    def remove_outliers(self,data:pd.DataFrame):
        quantile_25=data['Item_Visibility'].quantile(0.25)
        quantile_75=data['Item_Visibility'].quantile(0.75)
        IQR=quantile_75-quantile_25
        lower_limit=quantile_25-1.5*IQR
        upper_limit=quantile_75+1.5*IQR
        data=data[data['Item_Visibility']>lower_limit]
        data=data[data['Item_Visibility']<upper_limit]
        return data
    def drop_columns(self,data:pd.DataFrame):
        data=data.drop(['Item_Identifier'],axis=1)
        data=data.drop(['Outlet_Establishment_Year'],axis=1)
        return data
    def split_data(self,data:pd.DataFrame):
        X=data.drop('Item_Outlet_Sales',axis=1)
        Y=data['Item_Outlet_Sales']
        sc=StandardScaler()
        X=sc.fit_transform(X)
        x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=8)
        save_obj(self.config.scale_path,sc)
        return x_train,x_test,y_train,y_test
    def preprocess(self):
        data=self.load_data()
        logger.info("data loaded successfully")
        self.treat_null_values(data=data)
        logger.info("Treated null values successfully")
        data['Item_Fat_Content'].replace({'LF':'Low Fat','low fat':'Low Fat','reg':'Regular'},inplace=True)
        data=self.treat_categorical_data(data)
        logger.info("Treated categorical values successfully")
        data=self.remove_outliers(data)
        logger.info("Removed outliers successfully")
        data=self.drop_columns(data)
        logger.info("dropped unnecessary columns successfully")
        x_train,x_test,y_train,y_test=self.split_data(data)
        logger.info("data splitting done successfully")
        save_obj(self.config.x_train_path,x_train)
        save_obj(self.config.x_test_path,x_test)
        save_obj(self.config.y_train_path,y_train)
        save_obj(self.config.y_test_path,y_test)
        logger.info("Object saved successfully")