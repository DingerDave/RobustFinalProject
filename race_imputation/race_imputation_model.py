import numpy as np
from typing import Union
import pandas as pd
import time
from collections import defaultdict
import copy
class RaceImputationModel():
    def __init__(self, input_cols, target_cols):
        self.input_cols = input_cols
        self.target_cols = target_cols
        self.interest_cols = input_cols + target_cols
        self.col_to_name_map = {col: self.interest_cols[col] for col in range(len(self.interest_cols))}
        self.name_to_col_map = {self.interest_cols[col]: col for col in range(len(self.interest_cols))}

    def _compute_counts_matrix(self, data):
        self.posterior_counts = defaultdict(lambda: 0)
        for row in data:
            indices = []
            
            # collect the indices for each target and input column
            for target_col in range(len(self.interest_cols)):
                indices.append(self.val_to_idx[target_col][row[target_col]])
            
            # increment the count for the indices
            self.posterior_counts[tuple(indices)] += 1
        
    def _compute_conditional_probabilities(self):
        input_cols = [self.name_to_col_map[col] for col in self.input_cols]
        self.joint_prob_counts = defaultdict(lambda: 0)
        self.conditional_probs = copy.deepcopy(self.posterior_counts)
        for key, val in self.posterior_counts.items():
            self.joint_prob_counts[tuple(key[col] for col in input_cols)] += val
        for key, val in self.posterior_counts.items():
            self.conditional_probs[key] /= self.joint_prob_counts[tuple(key[col] for col in input_cols)]

        
    def _fit(self, data: Union[pd.DataFrame, np.ndarray, dict]):

        if isinstance(data, np.ndarray):
            assert all([type(val) == int for val in self.interest_cols]), "Columns must be integers for numpy arrays"
            
        else:
            # reorder the columns to match the order of the input_cols and target_cols
            data = data[self.interest_cols].to_numpy()  
    
        # Get the unique entries of each input and target column    
        self.unique_vals = {col: np.unique(data[:, col]) for col in range(data.shape[1])}
        # Create a mapping from input/target value to index
        self.val_to_idx = {col: {val: idx for idx, val in enumerate(self.unique_vals[col])} for col in self.unique_vals.keys()}    
        self.col_to_val_map = {self.interest_cols[col]: {val: [col, idx] for idx, val in enumerate(self.unique_vals[col])} for col in self.val_to_idx.keys()}
        
        self._compute_counts_matrix(data)
        self._compute_conditional_probabilities()

