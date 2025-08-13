from typing import List

import pandas as pd


class CFG:
    train_csv = './data/train.csv'
    test_csv = './data/test.csv'
    gdp_csv = './data/gdp.csv'

    countries = ['Canada', 'Finland', 'Italy', 'Kenya', 'Norway', 'Singapore']
    stores = ['Discount Stickers', 'Stickers for Less', 'Premium Sticker Mart']
    products = ['Holographic Goose', 'Kaggle', 'Kaggle Tiers', 'Kerneler', 'Kerneler Dark Mode']
    years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]

    worldbank_api_url = 'https://api.worldbank.org/v2/country/{0}/indicator/NY.GDP.PCAP.CD?date={1}&format=json'
    alpha3 = {'Finland': 'FIN', 'Canada': 'CAN', 'Italy': 'IT', 'Kenya': 'KEN', 'Singapore': 'SGP', 'Norway': 'NOR'}

    countries_2l = {'Finland': 'FI', 'Canada': 'CA', 'Italy': 'IT', 'Kenya': 'KE', 'Singapore': 'SG', 'Norway': 'NO'}
    holiday_response_len = 10

    country_factor_processing = 'const'