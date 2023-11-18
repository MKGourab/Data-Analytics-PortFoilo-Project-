import os
import sys
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
from src.exception import CustomException
from src.logger import logging

import warnings
warnings.filterwarnings("ignore")


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        results = {}

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, zero_division=1)
            confusion_mat = confusion_matrix(y_test, y_pred)
            
            # Calculate precision, recall, and f1 for each class
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

            results[model_name] = {
                'accuracy': accuracy * 100,
                'confusion_matrix': confusion_mat,
                'classification_report': class_report,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

        return results

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)