from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

 

if __name__=='__main__':
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    tranform=DataTransformation()
    train_arr,test_arr,m=tranform.initaite_data_transformation(train_data,test_data)
    trainer=ModelTrainer()
    trainer.initate_model_training(train_arr,test_arr)