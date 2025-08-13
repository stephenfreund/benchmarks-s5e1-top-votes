from datetime import date
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge

import holidays


def get_country_holidays(country_2l: str, years: List[int]) -> pd.DataFrame:
    hds_items = holidays.country_holidays(country_2l[:2], years=years).items()
    hds = pd.DataFrame(hds_items, columns=['date', 'name'])
    for ind, (date, name) in hds[hds['name'].str.contains('; ')].iterrows():
        for hname in name.split('; '):
            hds.loc[len(hds)] = date, hname#{'date': date, 'name': hname}
        hds.drop(ind, inplace=True)
    if country_2l == 'KE':
        hds_kenya = pd.read_csv('./data/hds_kenya.csv', parse_dates=['date'])
        hds_kenya = hds_kenya[hds_kenya.date.dt.year.isin(years)]
        hds_kenya.date = hds_kenya.date.dt.date
        hds = pd.concat((hds, hds_kenya)).sort_values(by='date')
        hds.loc[hds.name == 'Feast of the ', 'name'] = 'Feast of the Sacrifice'
        hds.loc[hds.name == 'Kenyatta Day', 'name'] = 'Mashujaa Day'
    if country_2l == 'SG_splitNY':
        hds['date1'] = pd.DatetimeIndex(hds.date)
        hds['year'] = hds.date1.dt.year

        dfm = hds[hds.name == 'Chinese New Year'].groupby('year').date1.min()
        for dt in dfm.values:
            hds.loc[(hds.name == 'Chinese New Year') & (hds.date1 == dt), 'name'] = 'Chinese New Year 1'
        hds.drop(['year', 'date1'], axis=1, inplace=True)
    return hds


def show(df: pd.DataFrame) -> None:
    countries = df.country.unique()
    years = df.year.unique()

    fig, ax = plt.subplots(nrows=len(countries))
    fig.set_size_inches(24, 3 * len(countries))
    colors = 'rgbcmkyrgbcmky'
    for n, country in enumerate(countries):
        ax[n].set_title(country)
        for m, y in enumerate(years):
            dt = df[(df.country == country) & (df.year == y)].groupby('dayofyear').total.mean()
            ax[n].plot(dt, colors[m], label=y)
            dt = df[(df.country == country) & (df.year == y)].groupby('dayofyear').holiday_flag.mean()
            ax[n].plot(0.6 + 0.35 * dt, colors[m])
        ax[n].legend()
    plt.show()


def insert_holiday_flag(df: pd.DataFrame,
                        countries_2l: Dict[str, str],
                        years: List[int],
                        holiday_response_len: int = 10) -> None:
    df['holiday_flag'] = 0
    df['holiday_period_flag'] = 0

    for country in df.country.unique():
        hds = get_country_holidays(countries_2l[country], years=years)
        for holiday in hds.date:  # , _ in holidays.country_holidays(countries_2l[country], years=years).items():
            period1 = pd.date_range(holiday, periods=1)
            period2 = pd.date_range(holiday, periods=holiday_response_len)
            df.loc[(df.country == country) & (df.date.isin(period1)), 'holiday_flag'] = 1
            df.loc[(df.country == country) & (df.date.isin(period2)), 'holiday_period_flag'] = 1


def show_response(df: pd.DataFrame,
                  countries_2l: Dict[str, str],
                  save_to_df: bool = False,
                  verbose: bool = False) -> None:
    countries = df.country.unique()
    years = df.year.unique()
    first_year, last_year = min(years), max(years)

    # Map function to extract the day of the data series beginning
    def datediff_firstday(date_input: pd.DatetimeIndex) -> int:
        return (date_input - date.fromisoformat(f'{first_year}-01-01')).days

    # The response function of each holiday
    def response_form(t: np.ndarray) -> np.ndarray:
        return np.exp(-(t - 4.5) ** 2 / 2 / 4)

    response = response_form(np.arange(10))

    if verbose:
        fig, ax = plt.subplots(nrows=len(countries))
        fig.set_size_inches(24, 3 * len(countries))

    for n, country in enumerate(countries):
        # We download the holidays data, adding 2009 year data
        hds = get_country_holidays(countries_2l[country], years=range(first_year - 1, last_year + 1))

        # Don't know what this is, but it helped in pgs319
        hds.loc[len(hds), :] = (date.fromisoformat(f'{first_year - 1}-12-28'), 'BEGIN')
        # hds.loc[len(hds), :] = (date.fromisoformat('2021-12-26'), 'CH1')
        # hds.loc[len(hds), :] = (date.fromisoformat('2022-01-01'), 'CH2')

        # daynum may be less than zero if in 2009
        hds['daynum'] = hds.date.map(datediff_firstday)
        d1 = (date.fromisoformat(f'{last_year}-12-31') - date.fromisoformat(f'{first_year}-01-01')).days
        d2 = (date.fromisoformat(f'{first_year - 1}-01-01') - date.fromisoformat(f'{first_year}-01-01')).days

        # Create data arrays with responses for each holiday name
        hd_data = []
        for hname in hds.name.unique():
            dt = np.zeros(d1 - d2)
            dt[hds[hds['name'] == hname].daynum - d2] = 1
            dt = np.convolve(response, dt)
            hd_data.append(dt)

        # Select data to fit and predict
        x = np.array(hd_data)[:, -d2:d1 - d2 + 1].T
        y = df[df.country == country].groupby('daynum').total.mean()

        fit_indices = ~pd.isna(y)

        # Make prediction
        reg = Ridge()
        reg.fit(x[fit_indices], y[fit_indices])
        holiday_factor = reg.predict(x)

        if save_to_df:
            df.loc[df.country == country, 'holiday_factor'] =\
                df[df.country == country].daynum.map(dict(zip(range(d1 + 1), holiday_factor)))

        if verbose:
            ax[n].plot(y, label='Total sales')
            ax[n].plot(y / holiday_factor, label='Reminder')
            ax[n].set_title(country)
            ax[n].legend()
    if verbose:
        plt.show()


def get_holidays(country_2l: str) -> pd.DataFrame:
    hds = get_country_holidays(country_2l, years=range(2009, 2020))
    hds['doy'] = pd.DatetimeIndex(hds.date).dayofyear

    if country_2l == 'Argentina':
        hds.loc[hds.name == 'Good Friday; Veterans Day and the Fallen in the Malvinas War', 'name'] = 'Good Friday'
        # It is possible also to add 'Veterans Day and the Fallen in the Malvinas War', but I skip it for a while

        # Rename a holiday that happened only once in 2022
        hds.loc[hds.name == 'National Census Day 2022', 'name'] = 'Bridge Public Holiday'

    if country_2l == 'Canada':
        pass

    if country_2l == 'Estonia':
        pass

    if country_2l == 'Japan':
        pass

    if country_2l == 'Spain':
        # Add new holidays in Spain, which absent in holidays library
        hds.loc[len(hds)] = [date.fromisoformat('2022-05-01'), 'Día del Trabajador', 121]
        hds.loc[len(hds)] = [date.fromisoformat('2022-12-25'), 'Navidad', 359]

    hds.name = hds.name.apply(lambda x: x.replace(' (Trasladado)', ''))
    hds.name = hds.name.apply(lambda x: x.replace(' (Observed)', ''))

    hds.sort_values('date')
    return hds


def get_holiday_factor(df: pd.DataFrame, hds: pd.DataFrame, ts_to_fit: np.ndarray, verbose: bool = True):
    years = df.year.unique()
    first_year, last_year = min(years), max(years)

    # Map function to extract the day of the data series beginning
    def datediff_firstday(date_input: pd.DatetimeIndex) -> int:
        return (date_input - date.fromisoformat(f'{first_year}-01-01')).days

    # The response function of each holiday
    def response_form(t: np.ndarray, t0: float, sigma0: float) -> np.ndarray:
        return np.exp(-((t - t0) / sigma0) ** 2 / 2)

    # Don't know what this is, but it helped in pgs319
    hds.loc[len(hds), :] = (date.fromisoformat(f'{first_year - 1}-12-28'), 'BEGIN', 363)

    # Daynum may be less than zero if in the first year
    hds['daynum'] = hds.date.map(datediff_firstday)
    d1 = (date.fromisoformat(f'{last_year}-12-31') - date.fromisoformat(f'{first_year}-01-01')).days
    d2 = (date.fromisoformat(f'{first_year - 1}-01-01') - date.fromisoformat(f'{first_year}-01-01')).days

    # Create data arrays with responses for each holiday name
    hnames = hds.name.unique()
    t0s = {hname: 4.5 for hname in hnames}
    sigma0s = {hname: 2.0 for hname in hnames}

    score = 0
    for test_hname in hnames:
        if verbose:
            print(test_hname)
        # We test different sigma0/t0 parameters

        for t0 in np.arange(4.0, 6.5, 0.25):
            for sigma0 in np.arange(0.1, 3.0, 0.1):
        # for t0 in [4.5, ]:
        #    for sigma0 in [2.0, ]:
                cur_t0s = t0s.copy()
                cur_sigma0s = sigma0s.copy()
                cur_t0s[test_hname] = t0
                cur_sigma0s[test_hname] = sigma0

                hd_data = []
                for hname in hnames:
                    dt = np.zeros(d1 - d2)
                    dt[hds[hds['name'] == hname].daynum - d2] = 1
                    response = response_form(np.arange(20), cur_t0s[hname], cur_sigma0s[hname])
                    dt = np.convolve(response, dt)
                    hd_data.append(dt)

                # Select data to fit and predict
                x = np.array(hd_data)[:, -d2:d1 - d2 + 1].T
                fit_indices = ~np.isnan(ts_to_fit)

                # Make prediction
                reg = Ridge()
                reg.fit(x[fit_indices], ts_to_fit[fit_indices])
                # reg.coef_[reg.coef_ < 0.01] = 0
                cur_score = reg.score(x[fit_indices], ts_to_fit[fit_indices])

                if cur_score > score:
                    score = cur_score
                    t0s[test_hname] = t0
                    sigma0s[test_hname] = sigma0
                    best_coef = reg.coef_
                    holiday_factor = reg.predict(x)
                    if verbose:
                        print('\t', t0, sigma0, score)

        if verbose:
            print(best_coef)
    return dict(zip(range(d1 + 1), holiday_factor))


def get_holiday_map(country_2l: str,
                    years: List,
                    strategy: Optional[Callable] = None,
                    response: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, List]:
    if response is None:
        response = np.exp(-(np.arange(10) - 4.5) ** 2 / 2 / 4)
    first_year, last_year = min(years), max(years)

    def datediff_firstday(date_input: pd.DatetimeIndex) -> int:
        return (date_input - date.fromisoformat(f'{first_year}-01-01')).days

    # We download the holidays data, adding 2009 year data
    hds = get_country_holidays(country_2l, years=range(first_year - 1, last_year + 1))

    # We apply external strategy to deal with observed holidays
    if strategy is not None:
        hds = strategy(hds)

    # Don't know what this is, but it helped in pgs319
    hds.loc[len(hds), :] = (date.fromisoformat(f'{first_year - 1}-12-28'), 'BEGIN')

    # daynum may be less than zero if in 2009
    hds['daynum'] = hds.date.map(datediff_firstday)
    d1 = (date.fromisoformat(f'{last_year}-12-31') - date.fromisoformat(f'{first_year}-01-01')).days
    d2 = (date.fromisoformat(f'{first_year - 1}-01-01') - date.fromisoformat(f'{first_year}-01-01')).days

    # Create data arrays with responses for each holiday name
    hd_data = []
    hds_name = hds.name.unique()
    for hname in hds_name:
        dt = np.zeros(d1 - d2)
        dt[hds[hds['name'] == hname].daynum - d2] = 1
        dt = np.convolve(response, dt)
        hd_data.append(dt)
    x = np.array(hd_data)[:, -d2:d1 - d2 + 1].T
    return np.arange(d1 + 1), x, hds_name


def strategy_observed(hds):
    hds1 = hds.copy()
    hds['year'] = pd.to_datetime(hds.date).dt.year
    for year in hds.year.unique():
        hdsy = hds[(hds['year'] == year)]
        observed = hdsy[hdsy['name'].str.endswith(' (observed)')].name.unique()
        for obs_name in observed:
            norm_name = obs_name[:-11]
            min_date = hdsy[hdsy['name'] == norm_name].date.min()
            hds1 = hds1.drop(hds1[(hds1.name == norm_name) & (hds1.date == min_date)].index)
    hds1.name = hds1.name.str.replace(' (observed)', '')
    return hds1


def strategy_normal(hds):
    return hds.drop(hds[hds.name.str.endswith(' (observed)')].index)


def strategy_union(hds):
    hds.name = hds.name.str.replace(' (observed)', '')
    return hds


def strategy_independent(hds):
    return hds
