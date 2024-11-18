import pandas as pd

def prep_voter_dataframe(voter_data, drop_null=True, shuffle=False):
    voter_data = voter_data[['last_name', 'first_name', 'middle_name', 'race', 'tract_code']]
    voter_data = voter_data.dropna() if drop_null else voter_data
    voter_data["tract_code"] = voter_data["tract_code"].astype(int).astype(str)
    voter_data["tract_code"] = voter_data["tract_code"].str.zfill(6)
    if shuffle:
        voter_data = voter_data.sample(frac=1).reset_index(drop=True)
    return voter_data

