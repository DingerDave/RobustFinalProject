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
    voter_data.loc[voter_data["race"]=="aian", "race"] = "asian"
    voter_data = voter_data.dropna() if drop_null else voter_data
    voter_data["tract_code"] = voter_data["tract_code"].astype(int).astype(str)
    voter_data["tract_code"] = voter_data["tract_code"].str.zfill(6)
    if shuffle:
        voter_data = voter_data.sample(frac=1).reset_index(drop=True)
    return voter_data

def prep_housing_dataframe(housing_data: pd.DataFrame, drop_null: bool=True, shuffle: bool=False) -> pd.DataFrame:
    """
    Prepares a housing dataframe by selecting specific columns, handling missing values,
    formatting the tract_code column, and optionally shuffling the data.

    Args:
        housing_data (pd.DataFrame): Input housing data.
        drop_null (bool, optional): If True, drops rows with null values. Defaults to True.
        shuffle (bool, optional): If True, shuffles the dataframe. Defaults to False.

    Returns:
        pd.DataFrame: The processed voter dataframe.
    """
    housing_data = housing_data[['derived_race', 'census_tract', 'derived_sex', 'denial_reason-1']]
    housing_data.rename(columns={'derived_race': 'race', "census_tract":"tract_code"}, inplace=True)
    housing_data = housing_data.dropna() if drop_null else housing_data
    housing_data.loc[housing_data["race"]=="White", "race"] = "white"
    housing_data.loc[housing_data["race"]=="Black or African American", "race"] = "black"
    housing_data.loc[housing_data["race"] == "Asian", "race"] = "asian"
    housing_data.loc[housing_data["race"] == "Free Form Text Only", "race"] = "other"
    # Extract the last 6 digits of the tract code
    housing_data["tract_code"] = housing_data["tract_code"].astype(int).astype(str).apply(lambda x: x[-6:])
    housing_data["tract_code"] = housing_data["tract_code"].str.zfill(6)
    housing_data["denied"] = housing_data["denial_reason-1"].apply(lambda x: 0 if x == 10 else 1)
    if shuffle:
        housing_data = housing_data.sample(frac=1).reset_index(drop=True)
    return housing_data