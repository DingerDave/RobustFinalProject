import sys
import os
import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from typing import Union
import pandas as pd
from collections import defaultdict
from race_imputation.race_imputation_model import RaceImputationModel

class DifferentialPrivacyRaceImputationModel(RaceImputationModel):
    """
    A model to impute missing target columns based on conditional probabilities
    derived from input columns, with differential privacy applied.
    """

    def __init__(self, input_cols, target_cols, epsilon=1.0):
        """
        Initialize the DifferentialPrivacyRaceImputationModel with input and target columns.

        Args:
            input_cols (list): List of input column names or indices.
            target_cols (list): List of target column names or indices.
            epsilon (float): The privacy parameter for differential privacy.
        """
        super().__init__(input_cols, target_cols)
        self.epsilon = epsilon  # privacy parameter

    def add_laplace_noise(self, value):
        """
        Add Laplace noise for differential privacy.

        Args:
            value (float): The value to which noise will be added.

        Returns:
            float: The value with added noise, ensuring non-negativity.
        """
        noise = np.random.laplace(0, value/self.epsilon) # scale is 1/value*epsilon
        # print(noise)
        return noise  # ensure non-negative counts

    def _compute_conditional_probabilities(self):
        """
        Compute conditional probabilities for the target columns based on the input columns,
        incorporating differential privacy.
        """
        input_cols = [self.name_to_col_map[col] for col in self.input_cols]
        self.joint_prob_counts = defaultdict(lambda: 0)
        self.conditional_probs = copy.deepcopy(self.posterior_counts)
    
        # Compute joint probabilities
        for key, val in self.posterior_counts.items():
            self.joint_prob_counts[tuple(key[col] for col in input_cols)] += val

        # Compute conditional probabilities
        for key, val in self.posterior_counts.items():
            self.conditional_probs[key] /= self.joint_prob_counts[tuple(key[col] for col in input_cols)]
            self.conditional_probs[key] += self.add_laplace_noise(1/self.joint_prob_counts[tuple(key[col] for col in input_cols)])
            self.conditional_probs[key] = max(0, self.conditional_probs[key])  # ensure non-negative probabilities

    def fit(self, data: Union[pd.DataFrame, np.ndarray, dict]):
        """
        Fit the model to the data by computing counts and probabilities with differential privacy.

        Args:
            data (Union[pd.DataFrame, np.ndarray, dict]): Input data.
        """
        super()._fit(data)  # call the original fit method
        #self._compute_conditional_probabilities()  # compute conditional probabilities with DP

    def predict(self, data: Union[pd.DataFrame, np.ndarray, dict], sampling_method=None, threshold=None):
        """
        Predict the target columns for the input data, using the DP conditional probabilities.

        Args:
            data (Union[pd.DataFrame, np.ndarray, dict]): The data to make predictions on.
            sampling_method (str, optional): The method to use for sampling the predictions.
                                           Options are "argmax", "threshold", or "sample".
            threshold (float, optional): The threshold to use for the "threshold" sampling method.

        Returns:
            np.ndarray: Data with predicted target columns.
        """
        return super()._predict(data, sampling_method, threshold)  # call the original predict method

    def demographic_parity(self) -> float:
        """
        Calculate the demographic parity of a model's predictions.
   
        Returns:
            float: The demographic parity.
        """
        return super().demographic_parity()