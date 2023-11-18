import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier, AdaBoostClassifier,GradientBoostingClassifier
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model
from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, 1:],          
                train_array[:, 0],          
                test_array[:, 1:],          
                test_array[:, 0]             
            )



            models={'Logistic Regression': LogisticRegression(max_iter=1000),
                    'Random Forest': RandomForestClassifier(),
                    'SVM': SVC(),
                    'Decission Tree': DecisionTreeClassifier(),
                    'KNN':KNeighborsClassifier(),
                    'Naive Bayes':GaussianNB(),
                    'Bagging': BaggingClassifier(),
                    'AdaBoost': AdaBoostClassifier(n_estimators= 501),
                    'Gradient Boosting':GradientBoostingClassifier(max_depth= 3, n_estimators= 500)
                    }
            
            model_results = evaluate_model(X_train, y_train, X_test, y_test, models)

            best_model_name = max(model_results, key=lambda k: model_results[k]['accuracy'])
            best_model_info = model_results[best_model_name]
            print("="*40)
            print("MODEL NAME\t\tMODEL ACCURACY")
            print("="*40)
            for model_name, model_info in model_results.items():
                print(f'{model_name} : {model_info["accuracy"]:.2f} %')
            
            print("="*40)      
            print(f'Best Model: {best_model_name}')
            print(f'Accuracy: {best_model_info["accuracy"]:.2f} %')
            print(f'Classification Report:\n{"="*65}\n{best_model_info["classification_report"]}\n{"="*65}\n')
            print(f'Confusion Matrix:\n{best_model_info["confusion_matrix"]}\n')

            # Logging
            logging.info(f'Model Report: {model_results}')
            logging.info(f'Best Model Found, Model Name: {best_model_name}, Accuracy Score: {round(best_model_info["accuracy"], 2)}%, Confusion Matrix:\n{best_model_info["confusion_matrix"]}')
            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=models[best_model_name]
            )
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)