import numpy as np, pandas as pd, matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as rc
from pylab import rcParams
from pandas.plotting import register_matplotlib_converters

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse

import plotly.express as px

from tqdm import tqdm

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly


class ProphetModel(BaseEstimator, TransformerMixin):

    def __init__(self, date_col, 
                 target_col, 
                 prophet_params= None, 
                 selected_cols= None, 
                 freq= '1min', 
                 f_period= 5):
        
        self.date_col= date_col
        self.target_col= target_col
        self.prophet_params = prophet_params or {}
        self.selected_cols= selected_cols
        self.freq= freq
        self.f_period= f_period

    def fit(self, X, y= None):
        
        if isinstance(X, pd.DataFrame):
            df= X.copy()
        else:
            df= pd.DataFrame(X)
        
        if not pd.api.types.is_datetime64_any_dtype(df[self.date_col]):
            df[self.date_col] = pd.to_datetime(df[self.date_col])

        feature_df= self.create_features(df, self.selected_cols)
        feature_df.set_index('ds', inplace= True)
        df_mod= self.feature_eng(feature_df[['ds', 'y']])

        self.m= Prophet(**self.prophet_params)
        feat_ls= [i for i in df_mod.columns if i not in ['ds', 'y']]
        for f in feat_ls:
            self.m.add_regressor(f)

        self.m.fit(df_mod)


        future= self.m.make_future_dataframe(periods= self.f_period, freq= self.freq)
        self.future_preds= self.feature_eng(future)


    def transform(self, X= None):
        self.forcast= self.m.predict(self.future_preds)
        return self.forcast

    def create_features(self, df: pd.DataFrame, selected_columns):
        print("all: ", df.columns.tolist())
        selected_columns= selected_columns or df.columns.tolist()
        df_selected = df.rename(
            columns={
                self.date_col: 'ds', 
                self.target_col: 'y'
            }
        )
        prophet_df= df_selected.sort_values(by= 'ds').reset_index(drop= True)
        self.display_df= prophet_df.copy()

        prophet_df['close_diff']= prophet_df.shift(1)['y']
        prophet_df['close_change'] = prophet_df['y'] - prophet_df['close_diff']
        prophet_df['close_change'].fillna(0, inplace=True)
        print("prophet_df: ", prophet_df.columns)
        rows= []
        for _, row in tqdm(prophet_df.iterrows(), total= prophet_df.shape[0]):
            row_data= dict(
                day_of_week= row.ds.dayofweek,
                day_of_month= row.ds.day,
                week_of_year= row.ds.week,
                month= row.ds.month,
                close_change= row.close_change,
                y= row.y,
                ds= row.ds
            )

            rows.append(row_data)
        _df= pd.DataFrame(rows)

        selected_columns= [sel_col for sel_col in selected_columns if sel_col not in ["ds", "close_change", "y"]]
        print("selected_columns: ", selected_columns)
        print("prophet_df_columns: ", prophet_df.columns.tolist())
        prophet_df= prophet_df[selected_columns]

        return pd.concat([prophet_df, _df], axis= 1)

    def feature_eng(self, data):
        data['quarter'] = data['ds'].dt.quarter
        data['month']=data['ds'].dt.month
        data['hour'] = data['ds'].dt.hour
        data['dayofmonth']=data['ds'].dt.day

        data['daysinmonth']=data['ds'].dt.daysinmonth
        data['dayofyear'] = data['ds'].dt.dayofyear

        return data

    def slide_display(self):
        plt_df = self.display_df.copy()
        fig = px.line(plt_df, x='ds', y='y', title='Slide Show')

        #slider
        fig.update_xaxes(
            rangeslider_visible = True,
            rangeselector = dict(
                buttons = list([
                    dict(count=1, label='1m', step="month", stepmode="backward"),
                    dict(count=2, label='2m', step="month", stepmode="backward"),
                    dict(count=2, label='5m', step="month", stepmode="backward")
                    ])
                )
            )

        return fig

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform()
    

    def plot_forcast(self):
        return self.m.plot(self.forcast)
    
    def plot_component(self):
        return self.plot_components(self.forcast)
    
    def _rms(self, test,):
        target= test[:self.f_period][self.target_col]
        pred= self.forcast["yhat"][-self.f_period:]

        return mse(target, pred, squared= False)