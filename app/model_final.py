import numpy as np
import joblib
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catboost import CatBoostRegressor, Pool
from data_utils import fillna_data
from utils.analysis_utils import model_analysis, model_visualization, discipline_analysis
from transform_utils import LightGBM_model
from params import FEATURES, CATBOOST_HYPERPARAMETRS, LIGHTGBM_HYPERPARAMETRS

class Trajectory: 
    def __init__(self):

        self.grade_model = CatBoostRegressor(**CATBOOST_HYPERPARAMETRS, verbose=False)
        self.type_model = LightGBM_model(**LIGHTGBM_HYPERPARAMETRS)

        self.features = FEATURES
        self.cat_features = None

    def _fillna_data(self, data):
        return fillna_data(data)
    
    def fit(self, data):
        data_filled = self._fillna_data(data)
        X = data_filled[self.features]
        y_grade = data_filled['target_grade']
        
        self.cat_features = [col for col in self.features if X[col].dtype in ('str', 'object')]
        X_pool = Pool(X, y_grade, self.cat_features)
        self.grade_model.fit(X_pool)

        self.type_model.fit(data_filled)

    def predict(self, data, fillna=True, visualization=False):
        if fillna==True:
            data = self._fillna_data(data)
        X = data[self.features]
        grades_pred = self.grade_model.predict(X)
        grades_pred = np.clip(np.round(grades_pred), a_min = 0, a_max = 10) # оценка от 0 до 10

        type_pred = self.type_model.predict_type(data)
        
        if visualization == True:
            self.visualize(grades_pred, type_pred)

        return grades_pred, type_pred 

    def analysis(self, y_true_grade, y_pred_grade, y_true_type, y_pred_type):
        return model_analysis(y_true_grade, y_pred_grade, y_true_type, y_pred_type)


    def visualize(self, grades_pred, type_pred):
        return model_visualization(grades_pred, type_pred)

    def discipline_check(self, data, discipline_name):
        discipline_analysis(data, discipline_name)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
    

    