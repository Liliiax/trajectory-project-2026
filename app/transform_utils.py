import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, TargetEncoder, OrdinalEncoder
import warnings
from sklearn.metrics import f1_score
for warn in [UserWarning, FutureWarning, RuntimeWarning]: warnings.filterwarnings("ignore", category = warn)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.params import FEATURES

    
class LightGBM_model: 
    
    def __init__(self, n_estimators,  max_depth, learning_rate, num_leaves, bagging_fraction, min_child_samples, feature_fraction):

        self.education_encoder = LabelEncoder()
        self.exam_encoder = LabelEncoder()
        self.absence_encoder = LabelEncoder()
        self.disc_encoder = TargetEncoder()
        self.program_encoder = TargetEncoder()
        
        self.place_encoder = OrdinalEncoder(
            categories=[['Без вступительных испытаний', 'Внеконкурсное поступление',
        'Бюджетные', 'Целевые', 'По межправительственным соглашениям',
        'Коммерческие за счет средств вуза', 'Коммерческие', 'Коммерческие места для иностранных граждан']])

        self.type_model = LGBMClassifier( n_estimators=n_estimators,max_depth=max_depth,
                                          learning_rate=learning_rate, num_leaves=num_leaves,
                                           bagging_fraction=bagging_fraction, min_child_samples=min_child_samples, feature_fraction=feature_fraction, random_state=42, verbose=-1)
        self.X_columns = FEATURES

    def fit_transform_data(self, data):

        data = data.copy()

        self.education_encoder.fit(['Бакалавриат', 'Магистратура'])
        data['education_level'] = self.education_encoder.transform(data['education_level'])

        self.exam_encoder.fit(['Первая сдача', 'Пересдача по уважительной причине', 'Пересдача', 'Пересдача с комиссией'])
        data['exam_type'] = self.exam_encoder.transform(data['exam_type'])
        data['exam_type_prev'] = self.exam_encoder.transform(data['exam_type_prev'])
        data['target_type'] = self.exam_encoder.transform(data['target_type'])

        self.absence_encoder.fit(['valid', 'invalid', '\\N'])
        data['absence_status'] = self.absence_encoder.transform(data['absence_status'])

        data['discipline'] = self.disc_encoder.fit_transform(data[['discipline']], data['target_grade'])

        data['program'] = self.program_encoder.fit_transform(data[['program']], data['target_grade'])

        data['place_type'] = self.place_encoder.fit_transform(data[['place_type']])

        return data


    def transform_data(self, data):

        data = data.copy()

        data['education_level'] = self.education_encoder.transform(data['education_level'])
        data['exam_type'] = self.exam_encoder.transform(data['exam_type'])
        data['exam_type_prev'] = self.exam_encoder.transform(data['exam_type_prev'])
        data['absence_status'] = self.absence_encoder.transform(data['absence_status'])
        data['discipline'] = self.disc_encoder.transform(data[['discipline']])
        data['program'] = self.program_encoder.transform(data[['program']])    
        data['place_type'] = self.place_encoder.transform(data[['place_type']])

        return data

    def fit(self, data):

        data = self.fit_transform_data(data)
        y_type = data['target_type']
        X = data[self.X_columns]
        self.type_model.fit(X, y_type) 
        
    def predict_type(self, data):

        data = self.transform_data(data)
        X = data[self.X_columns]
        pred = self.type_model.predict(X)
        return self.exam_encoder.inverse_transform(pred.astype(int))

    def predict_probabs(self, data):

        data = self.transform_data(data)
        X = data[self.X_columns]
        pred = self.type_model.predict_proba(X)

        classes = self.exam_encoder.inverse_transform(self.type_model.classes_.astype(int))

        return pd.DataFrame(pred, columns=classes)
    @property 
    def feature_importances_(self): 
        imp_sum = np.sum(self.type_model.feature_importances_) 
         
        return 100 * self.type_model.feature_importances_ / imp_sum
   

