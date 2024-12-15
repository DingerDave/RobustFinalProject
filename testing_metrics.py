import numpy as np
import pandas as pd
from metrics.fairness_metrics import demographic_parity, equal_opportunity, equal_odds
from data_preprocessing_utils import prep_voter_dataframe, prep_housing_dataframe

# load the data
nrows = 50000
voter_data_all = pd.read_csv('./data/combined_nc.csv', nrows=nrows)
housing_data_all = pd.read_csv('./data/hmda_nc.csv', nrows=nrows)

cleaned_housing_data = prep_housing_dataframe(housing_data_all, drop_null=True, shuffle=True)
