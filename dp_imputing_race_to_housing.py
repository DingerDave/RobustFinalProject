import numpy as np 
import pandas as pd
from differential_privacy.differential_privacy_model import DifferentialPrivacyRaceImputationModel
from race_imputation.race_imputation_model import RaceImputationModel
import data_preprocessing_utils
from metrics import fairness_metrics
import time

pd.set_option('display.max_columns', None)

voter_data_all = pd.read_csv('./data/combined_nc.csv')
voter_data = data_preprocessing_utils.prep_voter_dataframe(voter_data_all)

housing_data = pd.read_csv('./data/hmda_nc.csv')
housing_data = data_preprocessing_utils.prep_housing_dataframe(housing_data)

def test_different_epsilons_and_thresholds(voter_data, housing_data, input_cols, target_cols, epsilon_values, thresholds):
    accuracies = {}
    accuracies_by_race = {}
    parities = {}

    threshold = 0.5

    for epsilon in epsilon_values:
        # use current epsilon
        if epsilon != 0:
            dp_imputation_model = DifferentialPrivacyRaceImputationModel(input_cols, target_cols, epsilon=epsilon)
        else:
            dp_imputation_model = RaceImputationModel(input_cols, target_cols)

        # fit model to voter data
        start = time.time()
        dp_imputation_model._fit(voter_data)
        print(f"Time to fit model with epsilon {epsilon}: ", time.time() - start)

        # predict on housing w argmax
        start = time.time()
        predicted_argmax = dp_imputation_model.predict(housing_data, sampling_method="argmax")
        print(f"Time to predict argmax with epsilon {epsilon}: ", time.time() - start)
        housing_data[f"pred_ri_argmax_epsilon_{epsilon}"] = predicted_argmax[:, 1]

        # predict on housing w sampling
        predicted_data_sample = dp_imputation_model.predict(housing_data, sampling_method="sample")
        housing_data[f"pred_ri_threshold_{threshold}_epsilon_{epsilon}"] = predicted_data_sample[:, 1]

        dp_housing_data_imputation_comp = housing_data[housing_data["race"] != "Race Not Available"]
        dp_housing_data_imputation_comp = dp_housing_data_imputation_comp [dp_housing_data_imputation_comp [f"pred_ri_threshold_{threshold}_epsilon_{epsilon}"] != "Null"]
        print(housing_data['race'].unique())
        print(housing_data[f"pred_ri_threshold_{threshold}_epsilon_{epsilon}"].unique())
        # accuracy_argmax = sum(dp_housing_data_imputation_comp[f"pred_ri_argmax_epsilon_{epsilon}"] == dp_housing_data_imputation_comp["race"]) / len(dp_housing_data_imputation_comp)
        # accuracy_sample = sum(dp_housing_data_imputation_comp[f"pred_ri_sample_epsilon_{epsilon}"] == dp_housing_data_imputation_comp["race"]) / len(dp_housing_data_imputation_comp)

        # # Initialize accuracies dict if first epsilon
        # if len(accuracies) == 0:
        #     accuracies = {
        #         'accuracy_argmax': [accuracy_argmax],
        #         'accuracy_sample': [accuracy_sample]
        #     }
        # else:
        #     accuracies['accuracy_argmax'].append(accuracy_argmax)
        #     accuracies['accuracy_sample'].append(accuracy_sample)

        # predict on housing w thresholds
        # for threshold in thresholds:
        #     predicted_data_threshold = dp_imputation_model.predict(housing_data, sampling_method="threshold", threshold=threshold)
        #     housing_data[f"pred_ri_threshold_{threshold}_epsilon_{epsilon}"] = predicted_data_threshold[:, 1]
            
        #     dp_housing_data_imputation_comp = housing_data[housing_data["race"] != "Race Not Available"]
        #     accuracy_threshold = sum(dp_housing_data_imputation_comp[f"pred_ri_threshold_{threshold}_epsilon_{epsilon}"] == dp_housing_data_imputation_comp["race"]) / len(dp_housing_data_imputation_comp)
            
        #     key = f'accuracy_threshold_{threshold}'
        #     if key not in accuracies:
        #         accuracies[key] = [accuracy_threshold]
        #     else:
        #         accuracies[key].append(accuracy_threshold)

        

        race_accuracies = fairness_metrics.accuracy_by_race(dp_housing_data_imputation_comp,
                                         f"pred_ri_threshold_{threshold}_epsilon_{epsilon}",
                                         "race",
                                         "race")
        
        # Initialize or append accuracies by race
        if len(accuracies_by_race) == 0:
            accuracies_by_race = {race: [acc] for race, acc in race_accuracies.items()}
        else:
            for race in race_accuracies:
                accuracies_by_race[race].append(race_accuracies[race])
       
        model = DifferentialPrivacyRaceImputationModel([f"pred_ri_threshold_{threshold}_epsilon_{epsilon}"], ["denied"])
        model.fit(dp_housing_data_imputation_comp)
        
        # Track parities
        parity = model.demographic_parity()
        print(parity)
        if len(parities) == 0:
            parities = {key: [value] for key, value in parity.items()}
        else:
            for key in parities:
                parities[key].append(parity.get(key, None))

    print(accuracies_by_race)

    print("/nparities/n")
    print(parities)

    # return accuracies, parities


    # save results 
    housing_data.to_csv('./data/hmda_nc_dp_ri_results.csv', index=False)
    print("Predictions saved to './data/hmda_nc_dp_ri_results.csv'")

    # for epsilon, accuracy in accuracies.items():
    #     print(f"Epsilon: {epsilon}")
    #     for k, v in accuracy.items():
    #         print(k, v)

    # for k, v in parities.items():
    #     print(k, v)

input_cols = ["tract_code"]
target_cols = ["race"]
# epsilon_values = np.arange(0.001, 0.01, 0.001)
# epsilon_values = [1, 2, 3, 4]
epsilon_values = range(1, 10, 1)
thresholds = []

test_different_epsilons_and_thresholds(voter_data, housing_data, input_cols, target_cols, epsilon_values, thresholds)

'''
error due to privacy < error due to race imputation
add privacy at individual level 
change epsilon values (0 - 1)
see how the accuracy of imputation changes
see how the test results change
'''