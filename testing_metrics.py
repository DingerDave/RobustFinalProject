import numpy as np
import pandas as pd
from metrics.fairness_metrics import demographic_parity, equal_opportunity, equal_odds
from data_preprocessing_utils import prep_voter_dataframe, prep_housing_dataframe
from race_imputation.race_imputation_model import RaceImputationModel

# load the data
nrows = 50000

housing_data_all = pd.read_csv('./data/hmda_nc_ri.csv', nrows=nrows)
housing_data_imputation_comp = housing_data_all[housing_data_all["race"]!= "Race Not Available"]
model = RaceImputationModel(["race"], ["denied"])
model._fit(housing_data_imputation_comp)
print(model.name_to_col_map)
print(model.demographic_parity())