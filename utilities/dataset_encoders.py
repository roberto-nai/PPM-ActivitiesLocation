import pandas as pd
import itertools

def index_encoding(data: pd.DataFrame, case_id_col: str, activity_col: str, eventnr_col: str, outcome_col: str) -> pd.DataFrame:
    """
    Encodes an event log dataset into a one-hot representation of activity sequences.

    Parameters:
        data (pd.DataFrame): The input event log dataframe containing cases, activities, and outcomes.
        case_id_col (str): Column name representing the unique identifier for each case.
        activity_col (str): Column name representing the activity or event name.
        eventnr_col (str): Column name representing the sequence number of the event within each case.
        outcome_col (str): Column name representing the outcome or label of each case.

    Returns:
        pd.DataFrame: A dataframe with one-hot encoded activity sequences for each case. Each row corresponds to a case, and columns represent encoded activities in each event step.
    """
    # Create a copy of the data to avoid modifying the original dataset
    data = data.copy()

    # Sort the data by case ID and event number
    indexed_data = data.sort_values([case_id_col, eventnr_col])
    
    # Ensure the activity column is a string and the event number column is an integer
    indexed_data[activity_col] = indexed_data[activity_col].astype(str)
    indexed_data[eventnr_col] = indexed_data[eventnr_col].astype(int)

    # Determine the maximum sequence length across all cases
    max_length = int(indexed_data.groupby(case_id_col)[activity_col].size().max())
    indexed_col = [f'e{i}' for i in range(1, max_length + 1)]
    
    # Create a pivot table to arrange activities in sequence order for each case
    all_event_nums = range(1, max_length + 1)
    encoded_data = pd.pivot_table(
        indexed_data,
        index=case_id_col,
        columns=eventnr_col,
        values=activity_col,
        aggfunc='first',  # Use the first value as aggregation
        fill_value=0
    ).reindex(columns=all_event_nums, fill_value=0)
    
    # Rename pivoted columns to match indexed column names
    encoded_data.columns = indexed_col

    # Retrieve unique outcomes and merge them into the encoded dataframe
    ID_Labels = indexed_data[[case_id_col, outcome_col]].drop_duplicates(subset=[case_id_col], keep='first')
    encoded_data = encoded_data.merge(ID_Labels.set_index(case_id_col), left_index=True, right_index=True)

    # Perform one-hot encoding for all activity columns at once
    one_hot_encoded_data = pd.concat(
        [pd.get_dummies(encoded_data[col], prefix=col) for col in indexed_col],
        axis=1
    )

    # Identify missing activity columns to ensure consistency
    all_activities = set(data[activity_col].unique())
    missing_cols = [
        f"{e}_{act}" for e, act in itertools.product(indexed_col, all_activities)
        if f"{e}_{act}" not in one_hot_encoded_data.columns
    ]
    
    # Add missing columns in a single operation
    missing_cols_data = pd.DataFrame(0, index=one_hot_encoded_data.index, columns=missing_cols)
    one_hot_encoded_data = pd.concat([one_hot_encoded_data, missing_cols_data], axis=1)

    # Drop the original indexed columns for event sequences
    encoded_data.drop(indexed_col, axis=1, inplace=True)

    # Concatenate the one-hot encoded columns to the main dataframe
    encoded_data = pd.concat([encoded_data, one_hot_encoded_data], axis=1)

    # Return the final encoded dataframe
    return encoded_data
