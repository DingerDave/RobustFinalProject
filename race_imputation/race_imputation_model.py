import numpy as np
from typing import Union
import pandas as pd
import time
from collections import defaultdict
import copy

class RaceImputationModel():
    """
    A model to impute missing target columns based on conditional probabilities
    derived from input columns.
    """

    def __init__(self, input_cols, target_cols, apply_dp=False):
        """
        Initialize the RaceImputationModel with input and target columns.

        Args:
            input_cols (list): List of input column names or indices.
            target_cols (list): List of target column names or indices.
        """
        self.input_cols = input_cols
        self.target_cols = target_cols
        self.apply_dp = apply_dp
        self.interest_cols = input_cols + target_cols
        self.col_to_name_map = {col: self.interest_cols[col] for col in range(len(self.interest_cols))}
        self.name_to_col_map = {self.interest_cols[col]: col for col in range(len(self.interest_cols))}
        self.conditional_probs = None

    def preprocess_data(self, data):
        """
        Preprocess the input data to ensure consistency.

        Args:
            data (Union[pd.DataFrame, np.ndarray]): Input data.

        Returns:
            np.ndarray: Processed data as a NumPy array.
        """
        if isinstance(data, np.ndarray):
            assert all([type(val) == int for val in self.interest_cols]), "Columns must be integers for numpy arrays"
        else:
            # Reorder the columns to match the order of the input_cols and target_cols
            data = data[self.interest_cols].to_numpy()
        return data

    def _compute_counts_matrix(self, data):
        """
        Compute the counts matrix for posterior probabilities.

        Args:
            data (np.ndarray): Preprocessed data.
        """
        self.posterior_counts = defaultdict(lambda: 0)
        for row in data:
            indices = []

            # Collect the indices for each target and input column
            for target_col in range(len(self.interest_cols)):
                indices.append(self.val_to_idx[target_col][row[target_col]])

            # Increment the count for the indices
            self.posterior_counts[tuple(indices)] += 1

    def _compute_conditional_probabilities(self):
        """
        Compute conditional probabilities for the target columns based on the input columns.
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

    def _fit(self, data: Union[pd.DataFrame, np.ndarray, dict]):
        """
        Fit the model to the data by computing counts and probabilities.

        Args:
            data (Union[pd.DataFrame, np.ndarray, dict]): Input data.
        """
        data = self.preprocess_data(data)

        # Get the unique entries of each input and target column
        self.unique_vals = {col: np.unique(data[:, col]) for col in range(data.shape[1])}

        # Create a mapping from input/target value to index
        self.val_to_idx = {col: {val: idx for idx, val in enumerate(self.unique_vals[col])} for col in self.unique_vals.keys()}
        self.col_to_val_map = {self.interest_cols[col]: {val: [col, idx] for idx, val in enumerate(self.unique_vals[col])} for col in self.val_to_idx.keys()}

        self._compute_counts_matrix(data)
        self._compute_conditional_probabilities()

    def _predict(self, data: Union[pd.DataFrame, np.ndarray, dict], sampling_method=None, threshold=None):
        """
        Predict the target columns for the input data.

        Args:
            data (Union[pd.DataFrame, np.ndarray, dict]): The data to make predictions on.
            sampling_method (str, optional): The method to use for sampling the predictions.
                                           Options are "argmax", "threshold", or "sample".
            threshold (float, optional): The threshold to use for the "threshold" sampling method.

        Returns:
            np.ndarray: Data with predicted target columns.
        """
        if self.conditional_probs is None:
            raise ValueError("Model must be fit before making predictions")

        if sampling_method is None:
            sampling_method = "argmax"
        elif sampling_method == "threshold":
            assert threshold is not None, "Must provide a threshold value for sampling method 'threshold'"

        data = self.preprocess_data(data)
        for row in data:
            indices = []
            for target_col in self.target_cols:
                input_cols = [self.name_to_col_map[col] for col in self.input_cols]
                input_vals = [row[col] for col in input_cols]

                # Handle missing or unknown values
                if any([val not in self.unique_vals[self.name_to_col_map[col]] for val, col in zip(input_vals, self.input_cols)]):
                    row[self.name_to_col_map[target_col]] = "Null"
                    continue

                indices = [self.val_to_idx[col][row[col]] for col in input_cols]
                conditional_probs = [
                    self.conditional_probs[tuple(indices + [val])] 
                    for val in self.val_to_idx[self.name_to_col_map[target_col]].values()
                ]

                if np.sum(conditional_probs) == 0:
                    row[self.name_to_col_map[target_col]] = "Null"
                    continue

                if sampling_method == "argmax":
                    row[self.name_to_col_map[target_col]] = self.unique_vals[self.name_to_col_map[target_col]][np.argmax(conditional_probs)]
                elif sampling_method == "threshold":
                    if max(conditional_probs) < threshold:
                        row[self.name_to_col_map[target_col]] = "Null"
                    else:
                        row[self.name_to_col_map[target_col]] = self.unique_vals[self.name_to_col_map[target_col]][np.argmax(conditional_probs)]
                elif sampling_method == "sample":
                    conditional_probs = conditional_probs / np.sum(conditional_probs)  # normalize probs
                    conditional_probs = np.nan_to_num(conditional_probs)  # replace nans
                    conditional_probs /= np.sum(conditional_probs)  # renormalize
                    row[self.name_to_col_map[target_col]] = np.random.choice(self.unique_vals[self.name_to_col_map[target_col]], p=conditional_probs)
                else:
                    raise ValueError("Invalid sampling method. Options are 'argmax', 'threshold', or 'sample'")
        return data


    def demographic_parity(self) -> float:
        """
        Calculate the demographic parity of a model's predictions.
   
        Returns:
            float: The demographic parity.
        """
        if self.conditional_probs is None:
            raise ValueError("Model must be fit before calculating fairness metrics")
        
        # get the highest probability based on the conditional probabilities
        conds = list(self.conditional_probs.items())
        # get only the first class
        first_class = conds[0][0][1]
        conds = [cond for cond in conds if cond[0][1] == first_class]
        probs = {cond[0][0]: cond[1] for cond in conds}
        max_prob = max(probs.values())
        col_to_race = {value: key for key, value in self.val_to_idx[0].items()}
        if "Null" in col_to_race:
            del col_to_race['Null']
        # print(probs.items())
        disparity = {col_to_race[col]: prob/max_prob for col, prob in probs.items()}
        
        return disparity