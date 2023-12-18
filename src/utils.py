import os
import sys
import pickle
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from src.exception import CustomException
from src.logger import logging

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
            # Convert sparse matrices to dense arrays
            X_train_dense = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
            X_test_dense = X_test.toarray() if hasattr(X_test, 'toarray') else X_test

        for model_name, model in models.items():
            model.fit(X_train_dense, y_train)
            y_pred = model.predict(X_test_dense)
            accuracy = accuracy_score(y_test, y_pred)
            classification_rep = classification_report(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)

            results[model_name] = {
                'model': model,
                'accuracy': accuracy*100,
                'classification_report': classification_rep,
                'confusion_matrix': conf_matrix
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