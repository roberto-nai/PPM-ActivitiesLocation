from pathlib import Path
import pandas as pd
from typing import Union

def dataset_stats(
    dataframe: pd.DataFrame,
    dataset_file_name: str,
    caseid_col: str,
    outcome_col: str,
    activity_col: str,
    dir_stats: str
) -> pd.DataFrame:
    """
    Analyse the dataframe based on case ID, outcome, and activity columns, including dataset name.
    
    Parameters:
    - dataframe (pd.DataFrame): The input dataframe.
    - dataset_file_name (str): The file name of the dataset being analysed.
    - caseid_col (str): Column name for case IDs.
    - outcome_col (str): Column name for outcome labels.
    - activity_col (str): Column name for activity labels.
    - dir_stats (str): Directory in which to save the stats.

    Returns:
    - pd.DataFrame: A single-row DataFrame with dataset name as the first column and all metrics as subsequent columns.
    """
    # Number of distinct case IDs
    distinct_cases = dataframe[caseid_col].nunique()

    # Group by case ID to calculate statistics for rows per case ID
    case_counts = dataframe.groupby(caseid_col).size()
    min_rows = case_counts.min()
    mean_rows = round(case_counts.mean(), 3)
    max_rows = case_counts.max()

    # Distinct outcome values
    distinct_outcomes = dataframe[outcome_col].unique()

    # Count and percentage of outcomes relative to distinct cases
    outcome_counts = dataframe.groupby(outcome_col)[caseid_col].nunique().astype(int)
    outcome_percentages = round((outcome_counts / distinct_cases) * 100, 3)

    # Number of distinct events (activities)
    distinct_events = dataframe[activity_col].nunique()

    # Create a single-row DataFrame with all metrics
    stats_dict = {
        "Dataset Name": dataset_file_name,
        "Distinct CaseIDs": distinct_cases,
        "Min Rows Per CaseID": min_rows,
        "Mean Rows Per CaseID": mean_rows,
        "Max Rows Per CaseID": max_rows,
        "Distinct Events (Activities)": distinct_events,
        "Distinct Outcomes": len(distinct_outcomes),
        "Outcome Values": str(list(distinct_outcomes)),
        "Outcome '0' Count": outcome_counts.get(0, 0),
        "Outcome '1' Count": outcome_counts.get(1, 0),
        "Outcome '0' Percentage": outcome_percentages.get(0, 0.0),
        "Outcome '1' Percentage": outcome_percentages.get(1, 0.0),
    }

    # Convert the dictionary to a single-row DataFrame
    stats_df = pd.DataFrame([stats_dict])

    # Generate file name and save the stats DataFrame
    file_stats = f"{Path(dataset_file_name).stem}_stats.csv"
    path_stats = Path(dir_stats) / file_stats
    stats_df.to_csv(path_stats, sep=";", index=False)

    return stats_df

def prefix_generator(df_in: pd.DataFrame, case_id_col: str, eventnr_col:str, prefix_dim: int) -> pd.DataFrame:
    """
    Generates a DataFrame containing at most `prefix_dim` rows for each unique value  in the `case_id_col`. 
    The number of rows retained for each case is inclusive of `prefix_dim`.

    Parameters:
    - df_in (pd.DataFrame): The input DataFrame containing the event log data.
    - eventnr_col (str): The column name of event number sequence. 
    - case_id_col (str): The name of the column representing case identifiers.
    - prefix_dim (int): The maximum number of rows (inclusive) to retain for each unique value in `case_id_col`.

    Returns:
    pd.DataFrame: A new DataFrame containing at most `prefix_dim` rows for each unique case in `case_id_col`. Rows are ordered by their occurrence within each case.
    """
    # List to collect prefixes
    prefixes_list = []

    # Iterate through each unique case
    for case in df_in[case_id_col].unique():
        # Filter DataFrame for the current case and sort by event number
        case_data = df_in[df_in[case_id_col] == case].sort_values(by=eventnr_col)

        # Determine the number of rows to include (up to and including `prefix_dim`)
        max_rows = min(len(case_data), prefix_dim)

        # Append only the rows up to the calculated limit
        prefixes_list.append(case_data.iloc[:max_rows])

    # Concatenate all prefixes into a single DataFrame
    result_df = pd.concat(prefixes_list, ignore_index=True)

    return result_df


