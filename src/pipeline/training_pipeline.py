from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

import warnings
warnings.filterwarnings('ignore') 

if __name__=='__main__':
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    tranform=DataTransformation()
    input_feature_train_arr,target_train_arr,input_feature_test_arr,target_test_arr,m=tranform.initaite_data_transformation(train_data,test_data)
    trainer=ModelTrainer()
    trainer.initate_model_training(input_feature_train_arr,target_train_arr,input_feature_test_arr,target_test_arr)