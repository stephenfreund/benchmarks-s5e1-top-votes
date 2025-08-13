from typing import Callable, Tuple

import pandas as pd


class CV:
    def __init__(self, metric: Callable, target_col: str, test_col: str = 'test', prediction_col: str = 'prediction'):
        self.metric = metric
        self.target_col = target_col
        self.test_col = test_col
        self.prediction_col = prediction_col

    def split(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df_cv = df[df[self.test_col] == 0].copy()
        df_cv[f'__{self.target_col}'] = df_cv[self.target_col]
        df_cv[self.test_col] = 0

        selection = True
        for col, val in kwargs.items():
            selection &= (df_cv[col] == val)

        df_cv.loc[selection, self.test_col] = 1
        df_cv.loc[selection, self.target_col] = None

        return df_cv

    def score(self, df: pd.DataFrame) -> Tuple[float, float]:
        train_true = df[(df[self.test_col] == 0) & ~pd.isna(df[f'__{self.target_col}'])][f'__{self.target_col}']
        train_pred = df[(df[self.test_col] == 0) & ~pd.isna(df[f'__{self.target_col}'])][self.prediction_col]
        test_true = df[(df[self.test_col] == 1) & ~pd.isna(df[f'__{self.target_col}'])][f'__{self.target_col}']
        test_pred = df[(df[self.test_col] == 1) & ~pd.isna(df[f'__{self.target_col}'])][self.prediction_col]

        train_score = None if len(train_true) == 0 else self.metric(train_true, train_pred)
        test_score = None if len(test_true) == 0 else self.metric(test_true, test_pred)

        return train_score, test_score
