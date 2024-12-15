import numpy as np 
import pandas as pd
from race_imputation import race_imputation_model
import data_preprocessing_utils
import time
# force all columns to show on head call
pd.set_option('display.max_columns', None)

voter_data_all = pd.read_csv('./data/combined_nc.csv', )
voter_data = data_preprocessing_utils.prep_voter_dataframe(voter_data_all)
imputation_model = race_imputation_model.RaceImputationModel(["tract_code"],["race"])

housing_data = pd.read_csv('./data/hmda_nc.csv',)
housing_data = data_preprocessing_utils.prep_housing_dataframe(housing_data)

start = time.time()
imputation_model._fit(voter_data)
print("Time to fit model: ", time.time() - start)
start = time.time()
predicted_argmax = imputation_model._predict(housing_data, sampling_method="argmax")
print("Time to predict argmax: ", time.time() - start)
housing_data["pred_ri_argmax"] = predicted_argmax[:,1]
predicted_data_sample = imputation_model._predict(housing_data, sampling_method="sample")
housing_data["pred_ri_sample"] = predicted_data_sample[:,1]
predicted_data_threshold_25 = imputation_model._predict(housing_data, sampling_method="threshold", threshold=.25)
housing_data["pred_ri_threshold_25"] = predicted_data_threshold_25[:,1]
predicted_data_threshold_5 = imputation_model._predict(housing_data, sampling_method="threshold", threshold=.5)
housing_data["pred_ri_threshold_5"] = predicted_data_threshold_5[:,1]
predicted_data_threshold_75 = imputation_model._predict(housing_data, sampling_method="threshold", threshold=.75)
housing_data["pred_ri_threshold_75"] = predicted_data_threshold_75[:,1]
predicted_data_threshold_95 = imputation_model._predict(housing_data, sampling_method="threshold", threshold=.95)
housing_data["pred_ri_threshold_95"] = predicted_data_threshold_95[:,1]

print("Time to predict: ", time.time() - start)
housing_data.to_csv('./data/hmda_nc_ri.csv', index=False)
