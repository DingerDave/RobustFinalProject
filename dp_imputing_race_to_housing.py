import numpy as np 
import pandas as pd
from differential_privacy.differential_privacy_model import DifferentialPrivacyRaceImputationModel
import data_preprocessing_utils
import time

pd.set_option('display.max_columns', None)

voter_data_all = pd.read_csv('./data/combined_nc.csv')
voter_data = data_preprocessing_utils.prep_voter_dataframe(voter_data_all)

housing_data = pd.read_csv('./data/hmda_nc.csv')
housing_data = data_preprocessing_utils.prep_housing_dataframe(housing_data)

def test_different_epsilons_and_thresholds(voter_data, housing_data, input_cols, target_cols, epsilon_values, thresholds):
    results = []

    for epsilon in epsilon_values:
        # use current epsilon
        dp_imputation_model = DifferentialPrivacyRaceImputationModel(input_cols, target_cols, epsilon=epsilon)

        # fit model to voter data
        start = time.time()
        dp_imputation_model.fit(voter_data)
        print(f"Time to fit model with epsilon {epsilon}: ", time.time() - start)

        # predict on housing w argmax
        start = time.time()
        predicted_argmax = dp_imputation_model.predict(housing_data, sampling_method="argmax")
        print(f"Time to predict argmax with epsilon {epsilon}: ", time.time() - start)
        housing_data[f"pred_ri_argmax_epsilon_{epsilon}"] = predicted_argmax[:, 1]

        # predict on housing w sampling
        predicted_data_sample = dp_imputation_model.predict(housing_data, sampling_method="sample")
        housing_data[f"pred_ri_sample_epsilon_{epsilon}"] = predicted_data_sample[:, 1]

        # predict on housing w thresholds
        for threshold in thresholds:
            predicted_data_threshold = dp_imputation_model.predict(housing_data, sampling_method="threshold", threshold=threshold)
            housing_data[f"pred_ri_threshold_{threshold}_epsilon_{epsilon}"] = predicted_data_threshold[:, 1]

    # save results 
    housing_data.to_csv('./data/hmda_nc_dp_ri_results.csv', index=False)
    print("Predictions saved to './data/hmda_nc_dp_ri_results.csv'")

input_cols = ["tract_code"]
target_cols = ["race"]
epsilon_values = [0.1, 0.5, 1.0, 2.0]
thresholds = [0.25, 0.5, 0.75, 0.95]

test_different_epsilons_and_thresholds(voter_data, housing_data, input_cols, target_cols, epsilon_values, thresholds)
