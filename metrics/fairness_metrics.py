import numpy as np
import pandas as pd

def demographic_parity(ri_model, demographic_col: str, prediction_col: str) -> float:
    """
    Calculate the demographic parity of a model's predictions.

    Args:
        demographic_cols (list[str]): The demographic columns to consider.
        predictions (np.ndarray): The model's predictions.
        

    Returns:
        float: The demographic parity.
    """
    ri_model.conditional_probs
        
def equal_opportunity(demographic_cols: list[str], predictions: np.ndarray, protected_attributes: pd.DataFrame, true_labels: np.ndarray) -> float:
    """
    Calculate the equal opportunity of a model's predictions.

    Args:
        demographic_cols (list[str]): The demographic columns to consider.
        predictions (np.ndarray): The model's predictions.
        protected_attributes (pd.DataFrame): The protected attributes.
        true_labels (np.ndarray): The true labels.

    Returns:
        float: The equal opportunity.
    """
    equal_opportunity = 0
    for col in demographic_cols:
        mask = protected_attributes[col].values == 1
        equal_opportunity += np.abs(predictions[mask] - true_labels[mask]).mean()
    return equal_opportunity / len(demographic_cols)


def equal_odds(demographic_cols: list[str], predictions: np.ndarray, protected_attributes: pd.DataFrame, true_labels: np.ndarray) -> float:
    """
    Calculate the equal odds of a model's predictions.

    Args:
        demographic_cols (list[str]): The demographic columns to consider.
        predictions (np.ndarray): The model's predictions.
        protected_attributes (pd.DataFrame): The protected attributes.
        true_labels (np.ndarray): The true labels.

    Returns:
        float: The equal odds.
    """
    equal_odds = 0
    for col in demographic_cols:
        mask = protected_attributes[col].values == 1
        equal_odds += np.abs(predictions[mask] - true_labels[mask]).mean()
        equal_odds += np.abs(predictions[~mask] - true_labels[~mask]).mean()
    return equal_odds / (len(demographic_cols) * 2)