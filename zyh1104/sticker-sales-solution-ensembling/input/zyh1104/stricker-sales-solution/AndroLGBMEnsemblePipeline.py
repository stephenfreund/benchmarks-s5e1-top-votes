import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import scipy.stats as stats
import json

class AndroLGBMEnsemblePipeline:
    def __init__(self, param_file, seed=42, n_splits=5, n_estimators=1000, early_stopping_rounds=100):
        self.SEED = seed
        self.n_splits = n_splits
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        
        # Load parameters from JSON file
        with open(param_file, 'r') as f:
            self.lgbm_params = json.load(f)
        self.lgbm_params['n_estimators'] = n_estimators  # Ensure this matches the class parameter
        self.lgbm_params['seed'] = seed  # Ensure this matches the class parameter

    def load_data(self, train_path, test_path):
        self.train_data = pd.read_csv(train_path)
        self.test_data = pd.read_csv(test_path)
        self.train_data = self.train_data.drop('id', axis=1)
        self.num_cols = list(self.train_data.select_dtypes(exclude=['object']).columns.difference(['num_sold']))
        self.cat_cols = list(self.train_data.select_dtypes(include=['object']).columns)
        
    def label_encode(self):
        self.label_encoders = {col: LabelEncoder() for col in self.cat_cols}
        for col in self.cat_cols:
            combined_data = pd.concat([self.train_data[col], self.test_data[col]])
            le = LabelEncoder()
            le.fit(combined_data)
            self.train_data[col] = le.transform(self.train_data[col])
            self.test_data[col] = le.transform(self.test_data[col])
            
    def plot_distributions(self):
        data = self.train_data['num_sold'].replace([np.inf, -np.inf], np.nan).dropna()
        params_norm = stats.norm.fit(data)
        params_lognorm = stats.lognorm.fit(data, floc=0)
        params_laplace = stats.laplace.fit(data)
        params_expon = stats.expon.fit(data)
        params_gamma = stats.gamma.fit(data, floc=0)

        def create_trace(data, dist_name, params):
            dist_func = getattr(stats, dist_name)
            x = np.linspace(min(data), max(data), 1000)
            y = dist_func.pdf(x, *params)
            hist = go.Histogram(x=data, nbinsx=50, histnorm='probability density', name=f'{dist_name.capitalize()} Histogram')
            pdf = go.Scatter(x=x, y=y, mode='lines', name=f'{dist_name.capitalize()} PDF')
            return [hist, pdf]

        traces = []
        traces.extend(create_trace(data, 'norm', params_norm))
        traces.extend(create_trace(data, 'lognorm', params_lognorm))
        traces.extend(create_trace(data, 'laplace', params_laplace))
        traces.extend(create_trace(data, 'expon', params_expon))
        traces.extend(create_trace(data, 'gamma', params_gamma))

        fig = go.Figure(data=traces)
        fig.update_layout(title='Distribution Visualizations', xaxis_title='Value', yaxis_title='Density')
        fig.show()

    def plot_eda(self):
        fig_box = px.box(self.train_data, y='num_sold', title='Box Plot of num_sold')
        fig_box.show()
        fig_violin = px.violin(self.train_data, y='num_sold', title='Violin Plot of num_sold')
        fig_violin.show()
        data = self.train_data['num_sold'].replace([np.inf, -np.inf], np.nan).dropna()
        fig_kde = ff.create_distplot([data], ['num_sold'], bin_size=0.5, show_rug=False)
        fig_kde.update_layout(title='Histogram with KDE', xaxis_title='Value', yaxis_title='Density')
        fig_kde.show()
        corr_matrix = self.train_data.corr()
        fig_heatmap = px.imshow(corr_matrix, text_auto=True, title='Correlation Heatmap')
        fig_heatmap.show()

    def mape(self, y_true, y_pred):
        return mean_absolute_percentage_error(np.expm1(y_true), np.expm1(y_pred))

    def train_model(self):
        self.train_data['num_sold'] = np.log1p(self.train_data['num_sold'])
        X = self.train_data.drop(['num_sold'], axis=1)
        y = self.train_data['num_sold']
        test = self.test_data.drop(['id'], axis=1)
        kf = KFold(self.n_splits, shuffle=True, random_state=self.SEED)
        scores1 = []
        test_preds1 = []

        for i, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            model = lgb.LGBMRegressor(**self.lgbm_params)
            callbacks = [lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False)]
            model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], 
                      categorical_feature=self.cat_cols, eval_metric='rmse', callbacks=callbacks)
            val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration_)
            score = self.mape(y_val_fold, val_pred)
            scores1.append(score)
            test_pred = np.maximum(model.predict(test, num_iteration=model.best_iteration_), 0)
            test_preds1.append(test_pred)
            print(f'LightGBM Fold {i + 1} mape: {score}')
        print(f'LightGBM mape: {np.mean(scores1):.5f}')

        y_preds = np.mean(test_preds1, axis=0)
        y_preds = np.expm1(y_preds)
        y_preds = y_preds * 1.01
        y_preds = np.clip(y_preds, 5, 5939)
        submission = pd.DataFrame({'id': self.test_data['id'], 'num_sold': y_preds})
        submission.to_csv('submission.csv', index=False)
        print("submission.csv successfully saved !!")
        print(submission.head())
