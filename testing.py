
import opendp.prelude as dp
dp.enable_features('contrib')
import numpy as np 
import pandas as pd
from race_imputation import race_imputation_model
import data_preprocessing_utils
import time
# force all columns to show on head call
pd.set_option('display.max_columns', None)

nrows = 500000
voter_data_all = pd.read_csv('./data/combined_nc.csv', nrows=nrows)
voter_data = data_preprocessing_utils.prep_voter_dataframe(voter_data_all,shuffle=True)
hold_out = int(len(voter_data)*.9)
imputation_model = race_imputation_model.RaceImputationModel(["last_name", "tract_code"],["race"])
start = time.time()
imputation_model._fit(voter_data[:hold_out])
print("Time to fit model: ", time.time() - start)
start = time.time()
predicted_data_argmax = imputation_model._predict(voter_data[hold_out:], sampling_method="argmax")
predicted_data_sample = imputation_model._predict(voter_data[hold_out:], sampling_method="sample")
predicted_data_threshold_25 = imputation_model._predict(voter_data[hold_out:], sampling_method="threshold", threshold=.25)
predicted_data_threshold_5 = imputation_model._predict(voter_data[hold_out:], sampling_method="threshold", threshold=.5)
predicted_data_threshold_75 = imputation_model._predict(voter_data[hold_out:], sampling_method="threshold", threshold=.75)
predicted_data_threshold_95 = imputation_model._predict(voter_data[hold_out:], sampling_method="threshold", threshold=.95)

print("Time to predict: ", time.time() - start)
# get the percentage of predictions that are "Null"

print("Percentage of predictions that are 'Null' argmax: ", np.sum(predicted_data_argmax[:,1] == "Null")/len(predicted_data_argmax))
print("Percentage that are not Null and correct", np.sum((predicted_data_argmax[:,1] != "Null") & (predicted_data_argmax[:,1] == voter_data.to_numpy()[hold_out:, 3]))/len(predicted_data_argmax))

print("Percentage of predictions that are 'Null' sample: ", np.sum(predicted_data_sample[:,1] == "Null")/len(predicted_data_sample))
print("Percentage that are not Null and correct", np.sum((predicted_data_sample[:,1] != "Null") & (predicted_data_sample[:,1] == voter_data.to_numpy()[hold_out:, 3]))/len(predicted_data_sample))

print("Percentage of predictions that are 'Null' threshold 25: ", np.sum(predicted_data_threshold_25[:,1] == "Null")/len(predicted_data_threshold_25))
print("Percentage that are not Null and correct", np.sum((predicted_data_threshold_25[:,1] != "Null") & (predicted_data_threshold_25[:,1] == voter_data.to_numpy()[hold_out:, 3]))/len(predicted_data_threshold_25))

print("Percentage of predictions that are 'Null' threshold 5: ", np.sum(predicted_data_threshold_5[:,1] == "Null")/len(predicted_data_threshold_5))
print("Percentage that are not Null and correct", np.sum((predicted_data_threshold_5[:,1] != "Null") & (predicted_data_threshold_5[:,1] == voter_data.to_numpy()[hold_out:, 3]))/len(predicted_data_threshold_5))

print("Percentage of predictions that are 'Null' threshold 75: ", np.sum(predicted_data_threshold_75[:,1] == "Null")/len(predicted_data_threshold_75))
print("Percentage that are not Null and correct", np.sum((predicted_data_threshold_75[:,1] != "Null") & (predicted_data_threshold_75[:,1] == voter_data.to_numpy()[hold_out:, 3]))/len(predicted_data_threshold_75))

print("Percentage of predictions that are 'Null' threshold 95: ", np.sum(predicted_data_threshold_95[:,1] == "Null")/len(predicted_data_threshold_95))
print("Percentage that are not Null and correct", np.sum((predicted_data_threshold_95[:,1] != "Null") & (predicted_data_threshold_95[:,1] == voter_data.to_numpy()[hold_out:, 3]))/len(predicted_data_threshold_95))