
import opendp.prelude as dp
dp.enable_features('contrib')
import numpy as np 
import pandas as pd
from race_imputation import race_imputation_model
import data_preprocessing_utils
import time
# force all columns to show on head call
pd.set_option('display.max_columns', None)


voter_data_all = pd.read_csv('./data/combined_nc.csv')
voter_data = data_preprocessing_utils.prep_voter_dataframe(voter_data_all)

imputation_model = race_imputation_model.RaceImputationModel(["last_name"],["race"])
start = time.time()
imputation_model._fit(voter_data)
print("Time to fit model: ", time.time() - start)



