from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from typing import List, Optional, Tuple

from sklearn.metrics import mean_absolute_percentage_error


def data_read_and_combine(train_csv: str, test_csv: str) -> pd.DataFrame:
    dfs = []
    for test, csv in enumerate((train_csv, test_csv)):
        df = pd.read_csv(csv, parse_dates=['date'])
        df['test'] = test
        dfs.append(df)
    return pd.concat(dfs)


def data_add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    # Date features
    df['year'] = df['date'].dt.year
    df['weekday'] = df['date'].dt.weekday
    df['dayofyear'] = df['date'].dt.dayofyear
    df['daynum'] = (df.date - df.date.iloc[0]).dt.days
    df['weeknum'] = df['daynum'] // 7
    df['month'] = df.date.dt.month

    # Sinusoidal features
    daysinyear = df.groupby('year').dayofyear.max().rename('daysinyear')
    df = df.join(daysinyear, on='year', how='left')
    df['partofyear'] = (df['dayofyear'] - 1) / df['daysinyear']
    df['partof2year'] = df['partofyear'] + df['year'] % 2
    df['partof3year'] = df['partofyear'] + df['year'] % 3

    df['sin 4t'] = np.sin(8 * np.pi * df['partofyear'])
    df['cos 4t'] = np.cos(8 * np.pi * df['partofyear'])
    df['sin 3t'] = np.sin(6 * np.pi * df['partofyear'])
    df['cos 3t'] = np.cos(6 * np.pi * df['partofyear'])
    df['sin 2t'] = np.sin(4 * np.pi * df['partofyear'])
    df['cos 2t'] = np.cos(4 * np.pi * df['partofyear'])
    df['sin t'] = np.sin(2 * np.pi * df['partofyear'])
    df['cos t'] = np.cos(2 * np.pi * df['partofyear'])
    df['sin t/2'] = np.sin(np.pi * df['partof2year'])
    df['cos t/2'] = np.cos(np.pi * df['partof2year'])

    df['sin t/3'] = np.sin(2 * np.pi * df['partof3year'] / 3)
    df['cos t/3'] = np.cos(2 * np.pi * df['partof3year'] / 3)

    return df


def get_gdp_factor(gdp_csv: str, countries: List[str], years: List[int]) -> pd.DataFrame:
    gdp = pd.read_csv(gdp_csv)
    gdp = gdp[(gdp['Country Name'].isin(countries))].set_index('Country Name')
    data = [(c, y, gdp.loc[c, str(y)]) for c in countries for y in years]
    gdp = pd.DataFrame(data, columns=['country', 'year', 'gdp_factor'])
    #gdp['gdp_factor'] = -17643.346899 + 85.42355636 * gdp['gdp_factor']

    # Relative GDP
    gdp_sum = gdp.groupby('year').gdp_factor.sum().rename('gdp_sum')
    gdp = gdp.join(gdp_sum, on='year', how='left')
    gdp['gdp_ratio'] = gdp['gdp_factor'] / gdp['gdp_sum']
    # gdp['gdp_factor'] = gdp['gdp_factor'] / gdp['gdp_sum']

    return gdp.set_index(['country', 'year'])


def get_store_factor(df: pd.DataFrame, exclude_countries: Optional[List[str]] = None) -> pd.DataFrame:
    exclude_countries = () if exclude_countries is None else exclude_countries
    df_no_can_ken = df[~df.country.isin(exclude_countries)]
    return df_no_can_ken.groupby(by='store').num_sold.mean().rename('store_factor').to_frame()


def insert_product_factor(df: pd.DataFrame,
                          exclude_countries: Optional[List[str]] = None,
                          sincos_features: Optional[List[str]] = None) -> None:

    exclude_countries = () if exclude_countries is None else exclude_countries
    sincos_features = ['sin t', 'cos t', 'sin t/2', 'cos t/2'] if sincos_features is None else list(sincos_features)

    df_no_can_ken = df[~df.country.isin(exclude_countries)]

    total = df_no_can_ken.groupby(by='date').num_sold.sum().rename('num_sold_total')
    df_no_can_ken = df_no_can_ken.join(total, on='date', how='left')
    df_no_can_ken['num_sold_ratio'] = df_no_can_ken['num_sold'] / df_no_can_ken['num_sold_total']

    df['product_factor'] = None
    for product in df['product'].unique():
        dt = df_no_can_ken[(df_no_can_ken['product'] == product) & (df_no_can_ken['test'] == 0)].groupby(by='date')
        x = dt[sincos_features].mean().to_numpy()
        y = dt.num_sold_ratio.sum().to_numpy()

        reg = Ridge(alpha=0)
        reg.fit(x, y)

        print(product, reg.intercept_, reg.coef_)

        df.loc[(df['product'] == product), 'product_factor'] = reg.predict(
            df.loc[(df['product'] == product), sincos_features].to_numpy())


def compare(df1: pd.DataFrame, df2: pd.DataFrame, columns: List[str]) -> None:
    if len(df1) != len(df2):
        print('Error: different size of two dataframes')
        return

    for column in columns:
        std = np.std(df1[column].to_numpy() - df2[column].to_numpy())
        stdp = std / np.std(df1[column])

        print(f'{column:24s}:\t\t{stdp * 100:.4f} %')


def get_weekday_factor(df: pd.DataFrame) -> pd.DataFrame:
    countries = df.country.unique()

    # A dataframe with index(weeknum, country) and columns(weekday) containing total sales
    num_sold_per_weeknum_country_weekday = df\
        .groupby(['weeknum', 'country', 'weekday'])['num_sold'].sum()\
        .reset_index().pivot(index=['weeknum', 'country'], columns='weekday')

    # A dataframe with index(weeknum, country) and columns(weekday) containing ratio of sales per each week day
    ratio_per_weeknum_country_weekday = num_sold_per_weeknum_country_weekday\
        .apply(lambda row: row / sum(row), axis=1).reset_index()

    # A zero-filled dataframe containing weekday factor for each country
    ratio_weekday = pd.DataFrame(columns=countries, data=np.zeros((7, len(countries))))

    # For each country and for each week day ...
    for n, country in enumerate(countries):
        for weekday in range(7):
            # Get the median value along the weeknum dimension
            ratio_weekday.loc[weekday, country] = ratio_per_weeknum_country_weekday\
                .loc[ratio_per_weeknum_country_weekday.country == country, ('num_sold', weekday)].median()

    ratio_weekday['mean'] = ratio_weekday.mean(axis=1)
    ratio_weekday['std'] = ratio_weekday.std(axis=1)

    # Important: we need to be sure, that std / mean is low, so the weekday factor actually works

    return ratio_weekday['mean'].rename('weekday_factor')


def get_dayofyear_factor(df: pd.DataFrame) -> pd.DataFrame:
    countries = df.country.unique()
    local_df = df.copy()

    factor = df['gdp_factor'] * df['product_factor'] * df['store_factor'] * df['weekday_factor']
    local_df['total'] = local_df['num_sold'] / factor

    data = pd.DataFrame()
    for country in countries:
        data[country] = local_df[(df.country == country) & (local_df.holiday_period_flag == 0)]\
            .groupby(['dayofyear']).total.median()
    data['median'] = data.median(axis=1)

    x = data.index.to_numpy()
    y = data['median'].to_numpy()

    def fourier(t: np.ndarray) -> np.ndarray:
        return np.array([np.sin(2*np.pi/365*t), np.cos(2*np.pi/365*t)])

    reg = Ridge(alpha=0.01)
    reg.fit(fourier(x).T, y.T)
    year_ratio = reg.predict(fourier(np.arange(1, 366)).T)
    year_ratio = np.append(year_ratio, year_ratio[-1])

    return pd.DataFrame({'dayofyear': np.arange(1, 367), 'dayofyear_factor': year_ratio}).set_index('dayofyear')


def get_sincos_factor(df: pd.DataFrame) -> pd.DataFrame:
    countries = df.country.unique()
    local_df = df.copy()

    factor = (df['gdp_factor'] * df['product_factor'] * df['store_factor'] * df['weekday_factor'] *
              df['dayofyear_factor'])
    local_df['total'] = local_df['num_sold'] / factor

    data = pd.DataFrame()
    for country in countries:
        data[country] = local_df[(df.test == 0) & (df.country == country) & (df.holiday_period_flag == 0)]\
            .groupby(['date']).total.median()
    data['median'] = data.median(axis=1)

    sincoscol = ['sin 4t', 'cos 4t', 'sin 3t', 'cos 3t', 'sin 2t', 'cos 2t', 'sin t', 'cos t', 'sin t/2', 'cos t/2']
    # sincoscol = ['sin 2t', 'cos 2t', 'sin t/2', 'cos t/2']
    # sincoscol = ['sin t/2', 'cos t/2']
    # sincoscol = ['sin t/2', 'cos t/2', 'sin t/3', 'cos t/3']

    # Linear regression on fourier series
    dfsc = local_df[local_df.test == 0].groupby('date')[sincoscol].mean()
    dfsc['median'] = data['median']

    x = dfsc[~pd.isna(dfsc['median'])][sincoscol].to_numpy()
    y = dfsc[~pd.isna(dfsc['median'])]['median'].to_numpy()

    reg = Ridge(alpha=0.01, fit_intercept=True)
    reg.fit(x, y)

    sincos_factor = reg.intercept_ + (df[sincoscol] * reg.coef_).sum(axis=1)

    return pd.DataFrame({'id': df['id'], 'sincos_factor': sincos_factor}).set_index('id')


def get_country_factor_const(df: pd.DataFrame, product: str) -> pd.DataFrame:
    factor = df['gdp_factor'] * df['product_factor'] * df['store_factor'] * df['weekday_factor'] \
             * df['dayofyear_factor'] * df['sincos_factor']
    local_df = df.copy()
    local_df['total'] = local_df['num_sold'] / factor
    country_factor = local_df[(local_df['product'] == product)].groupby('country').total.sum().rename('country_factor')
    return country_factor / country_factor.median()


def get_prediction(df: pd.DataFrame, factors: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    factor = 1
    for f in factors:
        factor *= df[f]

    total = df['num_sold'] / factor
    print('total median=', total.median())
    prediction = total.median() * factor
    return prediction, total


def get_mape(df: pd.DataFrame) -> float:
    return mean_absolute_percentage_error(df[(df.test == 0) & (~pd.isna(df.num_sold))].num_sold,
                                          df[(df.test == 0) & (~pd.isna(df.num_sold))].prediction)


def submit(df: pd.DataFrame, rounding: bool = True) -> None:
    submission = df[(df.test == 1)][['id', 'prediction']].rename(columns={'prediction': 'num_sold'})
    if rounding:
        submission['num_sold'] = submission['num_sold'].round().astype(int)

    submission.to_csv('submission.csv', index=False)
    print(submission.head())


def show_cps_factor(df: pd.DataFrame) -> None:
    countries = df.country.unique()
    stores = df.store.unique()
    products = df['product'].unique()
    years = df.year.unique()

    fig, ax = plt.subplots(ncols=3, nrows=2)
    fig.set_size_inches(24, 6)

    for n, (column, (values, filter_col, filter_val)) in enumerate({'country': (countries, 'product', 'Kaggle'),
                                                                    'store': (stores, 'country', 'Italy'),
                                                                    'product': (products, 'country', 'Italy')
                                                                    }.items()):
        ax[0][n].set_title(column)
        factor = {}
        for val in values:
            grp = df[(df[column] == val) & (df[filter_col] == filter_val)].groupby('year').total.median()
            factor[val] = grp.mean()
            ax[0][n].plot(years, grp, label=val)
            ax[1][n].plot(years, grp / grp.mean(), label=val)
        ax[0][n].legend()
        ax[1][n].legend()

        print(factor)

        df[f'{column[0]}_ratio'] = df[column].map(factor)

    plt.show()


def insert_country_factor_linear(df: pd.DataFrame,
                                 product: str,
                                 slope: bool = True,
                                 verbose: bool = False) -> pd.DataFrame:
    factor = df['gdp_factor'] * df['product_factor'] * df['store_factor'] * df['weekday_factor'] \
             * df['dayofyear_factor'] * df['sincos_factor']
    local_df = df.copy()
    local_df['total'] = local_df['num_sold'] / factor

    x = np.arange(0, df['daynum'].max() + 1).reshape(-1, 1)

    country_factor_median = []
    for country in local_df['country'].unique():
        cfc = local_df[(local_df['product'] == product) &
                       (local_df['country'] == country) &
                       (local_df['test'] == 0)].groupby('daynum').total.sum().mean()
        country_factor_median.append(cfc)
    country_factor_median = np.median(country_factor_median)

    if verbose:
        print('country_factor_median', country_factor_median)
        plt.figure()

    df['country_factor'] = None
    for country in local_df['country'].unique():
        cfc = local_df[(local_df['product'] == product) &
                       (local_df['country'] == country) &
                       (local_df['test'] == 0)].groupby('daynum').total.sum().mean()

        s = local_df[(local_df['product'] == product) &
                     (local_df['country'] == country) &
                     (local_df['test'] == 0)].groupby('daynum').total.sum()

        reg = Ridge(fit_intercept=slope)
        if slope:
            reg.fit(s.index.to_numpy().reshape(-1, 1), s)
            y = reg.predict(x)
        else:
            reg.fit(s.index.to_numpy().reshape(-1, 1) * 0 + 1, s)
            y = reg.predict(x * 0 + 1)

        factor = dict(zip(x.reshape(-1), y / country_factor_median))
        df.loc[(df.country == country), 'country_factor'] = df[(df.country == country)].daynum.map(factor)

        if verbose:
            plt.plot(s, label=country)
            plt.plot(x, y, 'k')
            plt.plot(x, x * 0 + cfc, 'm--')
            print(country, y[0] / country_factor_median)

    if verbose:
        plt.legend()
        plt.show()
