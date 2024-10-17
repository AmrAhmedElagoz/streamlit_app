columns_to_keep = ['EMERGENCY_INDICATOR', 'PROVIDER_DEPARTMENT_CODE', 'DOCTOR_CODE', 'PATIENT_AGE',
                   'PATIENT_NATIONALITY', 'PATIENT_GENDER', 'CLAIM_TYPE',
                   'UNIT_OF_AGE', 'TOTAL_CLAIMED_AMOUNT_SAR', 'TOTAL_DISCOUNT', 'TOTAL_DEDUCTIBLE',
                   'TOTAL_PATIENT_VATAMOUNT','TREATMENT_TYPE', 'PUR_NAME', 'CONTRACT_NO', 'CONTRACT_NAME',
                   'POLICY_NAME', 'NEW_BORN', 'NET_WITH_VAT', 'LINE_CLAIMED_AMOUNT_SAR', 'CO_INSURANCE',
                   'LINE_ITEM_DISCOUNT', 'NET_VAT_AMOUNT', 'PATIENT_VAT_AMOUNT', 'VAT_PERCENTAGE',
                   'QTY_STOCKED_UOM', 'UNIT_PRICE_STOCKED_UOM', 'UNIT_PRICE_NET', 'DISCOUNT_PERCENTAGE',
                   'CombinedText1', 'CombinedText2', 'CombinedText3', 'CombinedText4', 'CombinedText5',
                   'CombinedText6', 'CombinedText7', 'CombinedText8', 'CombinedText9', 'CombinedText10',
                   'CombinedText11', 'CombinedText12', 'CombinedText13', 'CombinedText14', 'CPG_COMPLIANCE',
                   'CombinedText15', 'CombinedText16', 'ICDText1', 'ICDText2', 'ICDText3', 'ICDText4',
                   'ICDText5', 'ICDText6', 'ICDText7', 'ICDText8', 'ICDText9', 'ICDText10', 'ICDText11',
                   'ICDText12', 'ICDText13', 'ICDText14', 'ICDText15', 'ICDText16', 'ComplaintText1',
                   'ComplaintText2', 'ComplaintText3', 'ComplaintText4', 'ComplaintText5', 'ComplaintText6',
                   'ComplaintText7', 'ComplaintText8', 'ComplaintText9', 'ComplaintText10', 'ComplaintText11',
                   'ComplaintText12', 'ComplaintText13','ComplaintText14', 'ComplaintText15', 'ComplaintText16']

_numerical_features = ['UNIT_OF_AGE', 'TOTAL_CLAIMED_AMOUNT_SAR', 'TOTAL_DISCOUNT', 'TOTAL_DEDUCTIBLE',
                      'TOTAL_PATIENT_VATAMOUNT', 'NET_WITH_VAT', 'LINE_CLAIMED_AMOUNT_SAR',
                      'CO_INSURANCE', 'LINE_ITEM_DISCOUNT', 'NET_VAT_AMOUNT', 'PATIENT_VAT_AMOUNT',
                      'QTY_STOCKED_UOM', 'UNIT_PRICE_STOCKED_UOM', 'UNIT_PRICE_NET',
                      'DISCOUNT_PERCENTAGE']

rejection_rate_cols= ['PROVIDER_DEPARTMENT_CODE', 'PUR_NAME', 'POLICY_NAME']
vats= {0.0: 0, 0.15: 1}

dropped_cols_to_train= ['OUTCOME', 'CONTRACT_NO', 'CONTRACT_NAME', 'VAT_PERCENTAGE']


from abc import ABC, abstractmethod

from utils import SkewnessTransformer, convert_numeric
from model_utils import grid_dict

import pickle
import pandas as pd, numpy as np
import json

from sklearn.metrics import recall_score
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score

from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor, XGBClassifier
from tqdm import tqdm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, SVR
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import xgboost as xgb

import lightgbm as lgb

import matplotlib
matplotlib.use('Agg')

models_dict= {'SVC': SVC(), 'LR': LogisticRegression(), 'Linear Regression': LinearRegression(),
              'ElasticNet': ElasticNet(), 'KNN_cls': KNeighborsClassifier(), 'KNN_reg': KNeighborsRegressor(),
              'RF_cls': RandomForestClassifier(), 'XGB_cls': XGBClassifier(), 'XGB_reg': XGBRegressor(),
              'Ridge': Ridge(), 'Lasso': Lasso(), 'extra_tree': ExtraTreesClassifier(), 'SVR': SVR(),
              'GradientBoosting_cls': GradientBoostingClassifier(), 'Adaboost': AdaBoostClassifier(),
              'DecisionTree_cls': DecisionTreeClassifier()}

normilizers= {'standard': StandardScaler(), 'min-max': MinMaxScaler(feature_range = (0, 1))}


class BaseModel(ABC):
    def __init__(self, algorithm, grid=False):
        self.pipeline = None

    @abstractmethod
    def build_pipeline(self, X, poly_feat=False, skew_fix=False):
        """This method must be implemented by subclasses and should set self.pipeline."""
        pass

    def ensure_pipeline_built(self):
        """Ensure the pipeline has been built by checking if self.pipeline is set."""
        if self.pipeline is None:
            raise ValueError("The pipeline has not been built. Ensure `build_pipeline` sets self.pipeline.")

    def fit(self, X, y):
        self.ensure_pipeline_built()
        self.pipeline.fit(X, y)

    def predict(self, X):
        self.ensure_pipeline_built()
        return self.pipeline.predict(X)


class KFoldModel(BaseEstimator, TransformerMixin):
    def __init__(self, model, n_splits=5, shuffle=True, random_state=None):
        self.model = model
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.cv = None
        self.scores_ = []
        self.best_estimators_ = []

    def fit(self, X, y):
        self.cv = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        self.scores_ = []
        self.best_estimators_ = []

        for train_index, val_index in tqdm(self.cv.split(X), total=self.n_splits, desc="K-Fold Progress"):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            self.model.fit(X_train, y_train)
            self.best_estimators_.append(self.model.best_estimator_)
            
            score = accuracy_score(y_val, self.model.predict(X_val))
            self.scores_.append(score)

        return self

    def predict(self, X):
        return self.best_estimators_[-1].predict(X)



class GridSearchModel(BaseEstimator, TransformerMixin):
    def __init__(self, alg, grid_params=None):
        self.alg = alg
        self.grid_params = grid_params if grid_params is not None else {}
        self.grid_search = None
        self.best_estimator_ = None

    def fit(self, X, y=None):
        self.grid_search = GridSearchCV(estimator=self.alg, param_grid=self.grid_params, cv=3)
        self.grid_search.fit(X, y)
        print('grid search applied')

        self.best_estimator_ = self.grid_search.best_estimator_
        return self
    
    def transform(self, X):
        return X
    
    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)


class Model:
    def __init__(self):
        self.pipeline = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.kmeans = KMeans(n_clusters=10, random_state=42)
        self.rejection_rate= {}
        self._y_test= pd.read_parquet('15OCT_y_test.parquet').reset_index().drop(columns = ['index'])
        self._y_train= pd.read_parquet('oct_y_train.parquet').reset_index().drop(columns = ['index'])

        self.algorithm= xgb.XGBClassifier(use_label_encoder=False, scale_pos_weight= 12,
                                        colsample_bytree= 0.5227358868077786, learning_rate= 0.02177698447071677,
                                        max_depth= 9, n_estimators= 952, lambda_l1= 3.9555068852216038, lambda_l2= 8.97246542262531,
                                        min_child_weight= 5, subsample= 0.7026980209228673, gamma= 2.7012741748633493)
                

    def build_pipeline(self, X, poly_feat=False, skew_fix=False):
        categorical_features = X.select_dtypes('object').columns.tolist()
        numerical_features = X.select_dtypes('number').columns.tolist() 

        categorical_transformer = Pipeline(steps=[])

        numerical_transformer = Pipeline(steps=[])

        if skew_fix:
            numerical_transformer.steps.append(('skew_fix', SkewnessTransformer(skew_limit=0.75))),
            numerical_transformer.steps.append(('num_imp', SimpleImputer(missing_values=np.nan, strategy="mean")))

        if poly_feat:
            numerical_transformer.steps.append(('Polynomial_Features', PolynomialFeatures(degree=2)))
            print('poly features applied')

        preprocessor = ColumnTransformer(transformers=[
            ('categorical', categorical_transformer, categorical_features),
            ('numerical', numerical_transformer, numerical_features)
        ])
            
        model_step = ('model', self.algorithm)

        self.pipeline = Pipeline(steps=[
            # ('preprocessor', preprocessor),
            model_step
        ])

    def reverse_label(self, y):
        classes = self.label_encoder.classes_
        encoded_values = self.label_encoder.transform(classes)
        d = {cls: enc for cls, enc in zip(classes, encoded_values)}

        return d, self.label_encoder.inverse_transform(y)

    def encode_label(self, df):
        with open("label_encoding_items.json") as f:
            d= json.load(f)

        to_keep= ['EMERGENCY_INDICATOR', 'PATIENT_NATIONALITY', 'PATIENT_GENDER', 
                  'CLAIM_TYPE', 'TREATMENT_TYPE', 'NEW_BORN']
        
        filtered_dict = {key: d[key] for key in to_keep if key in d}

        for col in filtered_dict:
            if col in df.columns:
                df[col] = df[col].map(filtered_dict[col])

        return df

    def label_age(self, ages):
        age_labels = []
        for age in ages:
            if age >= 0 and age <= 2:
                age_labels.append(1)
            elif age >= 3 and age <= 10:
                age_labels.append(2)
            elif age >= 11 and age <= 17:
                age_labels.append(3)
            elif age >= 18 and age <= 24:
                age_labels.append(4)
            elif age >= 25 and age <= 34:
                age_labels.append(5)
            elif age >= 35 and age <= 44:
                age_labels.append(6)
            elif age >= 45 and age <= 54:
                age_labels.append(7)
            elif age >= 55 and age <= 64:
                age_labels.append(8)
            else:
                age_labels.append(9)
                
        return age_labels

    def create_importance(self, importance_df):
        
        icd_importance_sum = 0
        for index, row in importance_df.iterrows():
            if row['Feature'].startswith('ICD'):
                icd_importance_sum += row['Importance']
                
        # Step 1: Calculate the total importance for features starting with 'ICD'
        icd_importance_sum = importance_df[importance_df['Feature'].str.startswith('ICD')]['Importance'].sum()

        # Step 2: Replace features that start with 'ICD' with 'ICD10'
        importance_df.loc[importance_df['Feature'].str.startswith('ICD'), 'Feature'] = 'DIAGNOSIS'

        # Step 3: Set the importance of 'ICD10' to the total sum
        importance_df.loc[importance_df['Feature'] == 'DIAGNOSIS', 'Importance'] = icd_importance_sum

        # If you want to keep only one row with 'ICD10' and its total importance:
        importance_df = importance_df.drop_duplicates(subset=['Feature'])
        
        service_importance_sum = 0
        for index, row in importance_df.iterrows():
            if row['Feature'].startswith('CombinedText'):
                service_importance_sum += row['Importance']
                
        # Step 1: Calculate the total importance for features starting with 'ICD'
        service_importance_sum = importance_df[importance_df['Feature'].str.startswith('CombinedText')]['Importance'].sum()

        # Step 2: Replace features that start with 'ICD' with 'ICD10'
        importance_df.loc[importance_df['Feature'].str.startswith('CombinedText'), 'Feature'] = 'SERVICE_DESCRIPTION'

        # Step 3: Set the importance_df of 'ICD10' to the total sum
        importance_df.loc[importance_df['Feature'] == 'SERVICE_DESCRIPTION', 'Importance'] = service_importance_sum

        # If you want to keep only one row with 'ICD10' and its total importance:
        importance_df = importance_df.drop_duplicates(subset=['Feature'])
        
        complaint_importance_sum = 0
        for index, row in importance_df.iterrows():
            if row['Feature'].startswith('ComplaintText'):
                complaint_importance_sum += row['Importance']
                
        # Step 1: Calculate the total importance for features starting with 'ICD'
        complaint_importance_sum = importance_df[importance_df['Feature'].str.startswith('ComplaintText')]['Importance'].sum()

        # Step 2: Replace features that start with 'ICD' with 'ICD10'
        importance_df.loc[importance_df['Feature'].str.startswith('ComplaintText'), 'Feature'] = 'SIGNS_AND_SYMPTOMS'

        # Step 3: Set the importance of 'ICD10' to the total sum
        importance_df.loc[importance_df['Feature'] == 'SIGNS_AND_SYMPTOMS', 'Importance'] = complaint_importance_sum

        # If you want to keep only one row with 'ICD10' and its total importance:
        importance_df = importance_df.drop_duplicates(subset=['Feature'])
        
        importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index().drop(columns = ['index'])
        
        return importance_df

    def _preprocessing(self, df: pd.DataFrame, y: pd.DataFrame, train= False):
        df= df[columns_to_keep]
        for feat in _numerical_features:
            df[feat] = df[feat].astype(float)

        provider_mapping = {provider: idx for idx, provider in enumerate(df['PROVIDER_DEPARTMENT_CODE'].unique(), start=1)}

        df['DOCTOR_CODE'] = df['PROVIDER_DEPARTMENT_CODE'].map(provider_mapping)
        df['OUTCOME'] = y['OUTCOME']

        if len(self.rejection_rate)== 0:
            for col in rejection_rate_cols:
                total_outcomes = df.groupby(col)['OUTCOME'].count()

                rejected_outcomes = df[df['OUTCOME'] == 0].groupby(col)['OUTCOME'].count()
                rejection_rate = round((rejected_outcomes / total_outcomes), 2)
                self.rejection_rate[col] = rejection_rate.fillna(0)
            
                df[col] = df[col].map(self.rejection_rate[col])

        else:
            # test data
            for col in rejection_rate_cols:
                df[col] = df[col].map(self.rejection_rate[col])

        df['SUB_ACCOUNT'] = df['POLICY_NAME']
        df.drop(columns= ['POLICY_NAME'], inplace= True)

        if train:
            X_vec = self.vectorizer.fit_transform(df['CONTRACT_NAME'].fillna(""))
            df['CONTRACT_NO (MEMBER_CLASS)'] = self.kmeans.fit_predict(X_vec)

        else:
            X_vec = self.vectorizer.transform(df['CONTRACT_NAME'].fillna(""))
            df['CONTRACT_NO (MEMBER_CLASS)'] = self.kmeans.predict(X_vec)            

        df['PATIENT_AGE'] = self.label_age(df['PATIENT_AGE'])
        df['VAT_EXISTENCE'] = df['VAT_PERCENTAGE'].map(vats)
        df['CPG_COMPLIANCE'] = df['CPG_COMPLIANCE'].map({'Compliant': 1, 'Not Compliant': 0})
        
        df.drop(columns = dropped_cols_to_train, inplace= True)
        df.rename(columns={'PUR_NAME': 'INSURANCE_COMPANY'}, inplace= True)
        return df


    def medical_rejection(self, df, y, df_test, y_test, train= False):
        df= convert_numeric(df)
        df_test= convert_numeric(df_test)
        y= convert_numeric(y)
        y_test= convert_numeric(y_test)

        df= self._preprocessing(df, y, train)
        df_test= self._preprocessing(df_test, y_test, train)
        train_rej = pd.concat([df, y], axis = 1)
        train_rej = train_rej[(train_rej['FINAL_OUTCOME'] == 0) | (train_rej['FINAL_OUTCOME'] == 2)]
        X_train_rej = train_rej.drop(columns = ['NOTES', 'OUTCOME' ,'NPHIES_LABEL', 'FINAL_OUTCOME',])
        y_train_rej = train_rej [['NOTES', 'OUTCOME', 'NPHIES_LABEL', 'FINAL_OUTCOME']]
        y_train_rej['FINAL_OUTCOME'] = y_train_rej['FINAL_OUTCOME'].map({2: 1, 0:0}).astype(int)

        # test_rej = pd.concat([df_test, y_test], axis = 1)
        # # test_rej = test_rej[(test_rej['FINAL_OUTCOME'] == 0) | (test_rej['FINAL_OUTCOME'] == 1)]
        # X_test_rej = test_rej.drop(columns = ['NOTES', 'OUTCOME',  'NPHIES_LABEL', 'FINAL_OUTCOME'])
        # y_test_rej = test_rej [['NOTES', 'OUTCOME', 'NPHIES_LABEL', 'FINAL_OUTCOME']]
        # y_test_rej['FINAL_OUTCOME'] = y_test_rej['FINAL_OUTCOME'].map({2: 1, 0:0}).astype(int)

        return X_train_rej, y_train_rej, df_test, y_test

    def train(self, X: pd.DataFrame, y: pd.Series, X_test, y_test):
        self.build_pipeline(X, skew_fix=False, poly_feat=False)
        _y= y['FINAL_OUTCOME']
        if _y.dtypes == 'object':
            _y = self.label_encoder.fit_transform(_y)


        X_train_rej, y_train_rej, X_test_rej, y_test_rej= self.medical_rejection(X,
                                                                                 y,
                                                                                 X_test,
                                                                                 y_test,
                                                                                 True)
        

        X_train_rej= self.encode_label(X_train_rej)
        X_test_rej= self.encode_label(X_test_rej)
        self.pipeline.set_params(
            model__eval_set=[(X_test_rej, y_test_rej['FINAL_OUTCOME'])],
            model__verbose=True,
            # model__early_stopping_rounds=20
        )
        self.pipeline.fit(X_train_rej, y_train_rej['FINAL_OUTCOME'])
        
        self.model = self.pipeline.named_steps['model']


    def preprocess(self, X, y, train):
        if train:
            X, y, _, _= self.medical_rejection(X, y, X, y)
        else:
            _, _, X, y= self.medical_rejection(X, y, X, y)
        X= self.encode_label(X)
        return X

    # def predict(self, X):
    #     X = self.preprocess(X)
    #     return self.model.predict(X)
    
    def predict_prob(self, X, random_number):
        y= self._y_test.loc[[random_number]]
        X= self.preprocess(X, y, train= False)
        return self.model.predict_proba(X)
    
    def preprocess_for_shap(self, X, train= True, random_number= None):
        if train:
            X= self.preprocess(X, self._y_train, train= True)
        else:
            X= self.preprocess(X, self._y_test, train= False)

        if random_number is not None:
            y= self._y_test.iloc[[random_number]]
            X= self.preprocess(X, y, train= False)
        
        return X

    def cls_metrics(self, X, y_true):
        y_pred = self.predict(X)
        if y_true.dtypes == 'object':
            y_pred = self.label_encoder.inverse_transform(y_pred)

        cm = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)

        return cm, accuracy
    
    def reg_metrics(self, X, y_true):
        y_pred = self.predict(X)

        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return mse, r2

    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved successfully as: {file_path}")


def model(X_train, X_test, y_train, y_test):
    _model= Model()
    _model.train(X_train, y_train, X_test, y_test)
    print("model trained")
    if True:
        _model.save_model('claim_model.pkl')


def claim_inference(inf_df, X_train, X_test, y_train, y_test, random_number):
    try:
        with open('claim_model.pkl', 'rb') as f:
            _model= pickle.load(f)
            print("Model file found!")
            print("model loaded!")
    except:
        print("Model file not found, retraining...")
        model(X_train, X_test, y_train, y_test)
        print("finished training...")
        with open('claim_model.pkl', 'rb') as f:
            _model= pickle.load(f)
            print("model loaded!")

    return _model.predict_prob(inf_df, random_number)

def get_corresponding_labels(y, encode= False):
    try:
        with open('claim_model.pkl', 'rb') as f:
            _model= pickle.load(f)
        
        if encode:
            return _model.encode_label([y]) 
        
        return _model.reverse_label([y])
    except FileNotFoundError:
        print("Model file not found.")

    except pickle.UnpicklingError:
        print("Error loading the pickle model.")
