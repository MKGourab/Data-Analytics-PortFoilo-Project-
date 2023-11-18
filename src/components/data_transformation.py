import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer 
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')

            # Define transformers for categorical and numerical columns
            categorical_features = [
                'cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor',
                'gill_attachment', 'gill_spacing', 'gill_size', 'gill_color',
                'stalk_shape', 'stalk_root', 'stalk_surface_above_ring',
                'stalk_surface_below_ring', 'stalk_color_above_ring',
                'stalk_color_below_ring', 'veil_color', 'ring_number',
                'ring_type', 'spore_print_color', 'population', 'habitat'
                ]

            categorical_transformer = OneHotEncoder(sparse=False, drop='first')

            # Combine transformers for all features
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', categorical_transformer, categorical_features)
                ])

            return preprocessor

        except Exception as e:
            logging.error("Error in Data Transformation")
            raise CustomException(e, sys)
        
    def initaite_data_transformation(self, train_path, test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')
            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'class'
            drop_columns = [target_column_name]

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Transform using the preprocessor object
            logging.info("Applying preprocessing object on training and testing datasets.")

            # Print debug information
            logging.info(f"Columns in input_feature_train_df: {input_feature_train_df.columns}")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            le = LabelEncoder()
            target_feature_train_arr = le.fit_transform(target_feature_train_df)
            target_feature_test_arr = le.transform(target_feature_test_df)
          
            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[np.array(target_feature_train_arr), input_feature_train_arr]
            test_arr = np.c_[np.array(target_feature_test_arr), input_feature_test_arr]


            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.info("Exception occurred in the initiate_datatransformation")
            raise CustomException(e, sys)