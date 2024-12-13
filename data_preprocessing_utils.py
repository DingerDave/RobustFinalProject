import pandas as pd

def prep_voter_dataframe(voter_data: pd.DataFrame, drop_null: bool = True, shuffle: bool = False) -> pd.DataFrame:
    """
    Prepares a voter dataframe by selecting specific columns, handling missing values,
    formatting the tract_code column, and optionally shuffling the data.

    Args:
        voter_data (pd.DataFrame): Input voter data.
        drop_null (bool, optional): If True, drops rows with null values. Defaults to True.
        shuffle (bool, optional): If True, shuffles the dataframe. Defaults to False.

    Returns:
        pd.DataFrame: The processed voter dataframe.
    """
    voter_data = voter_data[['last_name', 'first_name', 'middle_name', 'race', 'tract_code']]
    voter_data = voter_data.dropna() if drop_null else voter_data
    voter_data["tract_code"] = voter_data["tract_code"].astype(int).astype(str)
    voter_data["tract_code"] = voter_data["tract_code"].str.zfill(6)
    if shuffle:
        voter_data = voter_data.sample(frac=1).reset_index(drop=True)
    return voter_data

def prep_housing_dataframe(housing_data: pd.DataFrame, drop_null: bool=True, shuffle: bool=False) -> pd.DataFrame:
    pass