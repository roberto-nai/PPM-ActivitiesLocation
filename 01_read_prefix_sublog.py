# IMPORTS
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split

### LOCAL IMPORT ###
from config import config_reader
from utilities.json_operations import extract_data_from_json
from utilities.dataset_encoders import index_encoding
from utilities.dataset_operations import dataset_stats, prefix_generator
from utilities.general_utilities import create_directory
from utilities.classification_models import ml_xgboost, ml_save_metrics

### GLOBALS ###
yaml_config = config_reader.config_read_yaml("config.yml", "config")
datasets_dir = str(yaml_config["DATASETS_DIR"])
datasets_config = str(yaml_config["DATASETS_CONFIG"])
datasets_num = int(yaml_config["DATASETS_NUM"])
csv_sep = str(yaml_config["DATASETS_DEFAULT_SEP"])
log_default_caseid = str(yaml_config["LOG_DEFAULT_CASEID"])
log_default_activity = str(yaml_config["LOG_DEFAULT_ACTIVITY"])
log_default_timestamp = str(yaml_config["LOG_DEFAULT_TIMESTAMP"])
log_default_eventnr = str(yaml_config["LOG_DEFAULT_EVENT_NR"])
log_default_outcome = str(yaml_config["LOG_DEFAULT_OUTCOME"])
log_default_threshold = int(yaml_config["LOG_DEFAULT_THRESHOLD"])
log_default_split = list(yaml_config["DEFAULT_SPLIT"])
encoding_type = str(yaml_config["ENCODING_TYPE"])
task_ml = dict(yaml_config["TASK_ML"])
task_script = dict(yaml_config["TASK_SCRIPT"])

# OUTPUT
datasets_encoded_dir = str(yaml_config["DATASETS_ENCODED_DIR"])
ml_results_dir = str(yaml_config["ML_RESULTS_DIR"])
datasets_stats_dir = str(yaml_config["DATASETS_STATS_DIR"])
datasets_prefix_dir = str(yaml_config["DATASETS_PREFIX_DIR"])
datasets_sublog_dir = str(yaml_config["DATASETS_SUBLOG_DIR"])

def main_pipeline(dataset_item:dict, pipeline_configuration:dict):
    print(">> Parsing dataset")

    print("-"*8)
    ### A dataset informations ###
    dataset_file_name = dataset_item['file_name']
    dataset_path = Path(datasets_dir) / dataset_file_name
    dataset_name = pipeline_configuration['dataset_name']

    ### Check column values from JSON settings ###
    csv_separator = dataset_item.get('separator') or csv_sep  # if empty, assign the default value
    case_id_col = dataset_item.get('case_id_column') or log_default_caseid
    activity_col = dataset_item.get('activity_column') or log_default_activity
    timestamp_col = dataset_item.get('timestamp_column') or log_default_timestamp
    eventnr_col = dataset_item.get('event_number_column') or log_default_eventnr
    outocome_col = dataset_item.get('outcome_column') or log_default_outcome
    outcome_value_1 = dataset_item.get('outcome_value_1') or 1
    outcome_value_0 = dataset_item.get('outcome_value_0') or 0
    prefix_list = dataset_item.get('prefix_list')
    prefix_list_len = len(prefix_list)

    ### Load the dataset ###
    print("> Reading:", dataset_path)
    print("CSV separator:", csv_separator)
    df_log = pd.read_csv(dataset_path, sep=csv_separator, low_memory=False)
    print("Dataframe shape:", df_log.shape)
    print("Distinct cases:", df_log[case_id_col].nunique())
    # print(df_log.info())  # debug
    print()

    ### Initial stats ###
    print("> Stats about the dataset read")
    df_dataset_stats = dataset_stats(
        df_log,
        dataset_file_name,
        case_id_col,
        outocome_col,
        activity_col,
        datasets_stats_dir,
    )
    print(df_dataset_stats.head())
    print()

    ### Removing cases with very low frequent activities ###
    print("> Removing cases with very low frequent activities")
    print("Threshold:", log_default_threshold)
    # Compute the frequency of activities for each case ID
    activity_frequency = df_log.groupby(case_id_col)[case_id_col].count()
    # Identify case IDs with low-frequency activities below the threshold
    low_frequency_cases = activity_frequency[
        activity_frequency < log_default_threshold
    ].index
    # Find case IDs associated with these low-frequency activities
    cases_to_remove = df_log[
        df_log[case_id_col].isin(low_frequency_cases)
    ][case_id_col].unique()
    print("Cases to be removed:", cases_to_remove.tolist())
    # Remove the identified case IDs from the log
    df_log = df_log[~df_log[case_id_col].isin(cases_to_remove)]
    print("Cases removed:", len(cases_to_remove))
    print()

    ### Refactoring data ###
    df_log[case_id_col] = df_log[case_id_col].astype(str)
    df_log[activity_col] = df_log[activity_col].str.lower()

    ### Cleaning ###
    df_log[activity_col] = df_log[activity_col].str.replace(" ", "")
    df_log[activity_col] = df_log[activity_col].str.replace("-", "")
    df_log[activity_col] = df_log[activity_col].str.replace("_", "")
    # print(df_log[activity_col].unique())  # debug
    df_log[activity_col] = df_log[activity_col].map(lambda x: x.split('-')[0])
    # print(df_log[activity_col].unique())  # debug

    ### Custom case for some specific dataset ###
    if 'lifecycle:transition' in df_log.columns:
        print("> Custom case 'lifecycle:transition'")
        df_log = df_log[df_log['lifecycle:transition'] == 'COMPLETE']
        df_log.drop(['lifecycle:transition'], axis=1, inplace=True)
        print()

    ### Ordering ###
    df_log.sort_values(
        [case_id_col, timestamp_col], ascending=[True, True], inplace=True
    )

    ### Add the event number, if needed ###
    if eventnr_col not in df_log.columns:
        print(f"> Adding column {eventnr_col}")
        df_log[eventnr_col] = df_log.groupby([case_id_col]).cumcount() + 1
        df_log.sort_values(
            [case_id_col, eventnr_col], ascending=[True, True], inplace=True
        )
        print()
    else:
        df_log.sort_values(
            [case_id_col, eventnr_col], ascending=[True, True], inplace=True
        )

    ### Outcome (label) ###
    print("> Outcome (label)")
    print("Column name:", outocome_col)
    print("Values before refactoring:", df_log[outocome_col].unique().tolist())
    df_log.loc[df_log[outocome_col] == outcome_value_1, outocome_col] = 1
    df_log.loc[df_log[outocome_col] == outcome_value_0, outocome_col] = 0
    print("Values after refactoring:", df_log[outocome_col].unique().tolist())
    print()

    ### Indexing ###
    print("> Indexing info")
    top_n = 5
    print(f"Top {top_n}")
    indexing_info = df_log.groupby(activity_col).agg({eventnr_col: ['mean', 'std']})
    indexing_info.columns = ['mean', 'std']
    indexing_info.reset_index(inplace=True)
    indexing_info.sort_values(by='mean', ascending=False, inplace=True)
    indexing_info.reset_index(inplace=True, drop=True)
    print(indexing_info.head(top_n))
    # print(f"Last {top_n}")
    # print(indexing_info.tail(top_n))
    print()

    ### Prefixes ###
    print("> Prefixes")
    sum_sublog = 0
    print(f"- Generating prefix of length {prefix_len}")
    df_prefix = prefix_generator(df_log, case_id_col, eventnr_col, prefix_len)
    file_name_p = f"{dataset_name}_P{prefix_len}.csv"
    path_prefix = Path(datasets_prefix_dir) / file_name_p
    # print(df_prefix.head())
    print("- Saving prefix dataset to:", path_prefix)
    df_prefix.to_csv(path_prefix, index=False, sep=",")

    ### Activity selection and sub-log ###
    print("- Activity selection and sublog")
    list_activities = df_prefix[activity_col].unique().tolist()  # Get the list of distinct activities inside the prefix
    list_activities_len = len(list_activities)
    # print("Distinct activities in this prefix (list):", list_activities)
    print("Distinct activities in this prefix (num):", list_activities_len)
    # For each prefix event log generated, create sublogs with traces containing one of the activities in the list
    j = 0
    for activity_name in list_activities:
        j += 1
        sum_sublog += 1
        print(f"Creating sublog with activity '{activity_name}'")

        # Identify the case IDs of cases that contain the specified activity
        case_ids_with_activity = df_prefix[df_prefix[activity_col] == activity_name][case_id_col].unique()
        # Filter the dataframe to include only rows belonging to the identified cases
        df_sublog = df_prefix[df_prefix[case_id_col].isin(case_ids_with_activity)]

        print(f"Sublog [{j}] shape:", df_sublog.shape)

        file_name_ps = f"{dataset_name}_P{prefix_len}_S{j}.csv"
        path_sublog = Path(datasets_sublog_dir) / file_name_ps
        print("Saving sublog dataset to:", path_sublog)
        df_sublog.to_csv(path_sublog, index=False, sep=",")

        ### For every trace, generate a factual and a counter-factual trace

    print("Prefixes generated for this dataset:", prefix_list_len)
    print("Sublogs generated for this dataset:", sum_sublog)
    print()

    ### Encoding ###
    if task_script["ENCODING"] == 1:
        print("> Encoding")
        df_log_encoded = None
        dataset_name_enc = f"{dataset_name}_{encoding_type}"  # Get the dataset name and add the encoding
        if encoding_type == "I":
            print("Index encoding")
            df_log_encoded = index_encoding(
                df_log, case_id_col, activity_col, eventnr_col, outocome_col
            )
            print(df_log_encoded.head(5))
            print("Dataframe encoded shape:", df_log_encoded.shape)
            file_name_enc = f"{dataset_name_enc}.csv"
            path_out = Path(datasets_encoded_dir) / file_name_enc
            print("Encoded data saved to:", path_out)
            df_log_encoded.to_csv(path_out, sep=csv_sep, index=False)
        print()

    ### Prediction ###
    if task_script["PREDICTION"] == 1:
        print("> Predictions")
        print("Outcome column name:", outocome_col)

        # Split the data into features and target (outcome / label)
        X = df_log_encoded.drop(columns=[outocome_col])  # Features (all except the outcome column)
        y = df_log_encoded[outocome_col]  # Target (outcome column)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        for key, value in task_ml.items():  # key = algorithm name, value = 1/0 (yes/no)
            if value == 1:
                print(f"Algorithm executed: {key}")
                metrics_model = ml_xgboost(
                    X_train, X_test, y_train, y_test, outcome_column=outocome_col
                )
                print("- Metrics")
                print(metrics_model)
                ml_save_metrics(
                    metrics_model, dataset_name_enc, ml_results_dir, key
                )
        print("-" * 8)


if __name__ == "__main__":
    print()
    print("*** PROGRAM START ***")
    print()

    ### Timing ###
    start_time = datetime.now().replace(microsecond=0)
    print("Start process:", str(start_time))
    print()

    ### Creating output directories ###
    print(">> Creating output directories")
    create_directory(datasets_encoded_dir)
    create_directory(ml_results_dir)
    create_directory(datasets_stats_dir)
    create_directory(datasets_prefix_dir)
    create_directory(datasets_sublog_dir)
    print()

    ### Read the JSON with datasets info ###
    print(">> Reading configuration file")
    path_json = Path(datasets_config)
    print("Path:", path_json)
    datasets_list = extract_data_from_json(path_json)
    datasets_list_len = len(datasets_list)
    print("Datasets found in JSON configuration:", datasets_list_len)
    print()

    ### If there is data in JSON, extract every dataset, every prefix and execute the pipeline ###
    if datasets_list_len > 0:
        i = 0
        for dataset_item in datasets_list:
            i+=1
            print(f"[{i}]")
            print(dataset_item)
            print()
            print("Dataset file:", dataset_item["file_name"])
            dataset_name = Path(dataset_item["file_name"]).stem
            prefix_list = dataset_item.get('prefix_list')
            prefix_list_len = len(prefix_list)
            print(f"Prefixes ({prefix_list_len}):", prefix_list)
            for prefix_len in prefix_list:
                ### Executing the main pipeline ###
                print(">> Executing pipeline")
                print("Dataset:", dataset_name)
                print("Prefix:", prefix_len)
                pipeline_configuration = {
                    "file_name": dataset_item["file_name"],
                    "dataset_name": dataset_name,
                    "train_val_test_split": dataset_item["train_val_test_split"],
                    "prefix_len": prefix_len
                }
                main_pipeline(dataset_item, pipeline_configuration)
            # Check how many datasets have still to be considered
            if datasets_num != 0 and i >= datasets_num:
                break
    else:
        print("ERROR! No configuration data (JSON) to be read, quitting the program.")

    end_time = datetime.now().replace(microsecond=0)
    print("End process:", str(end_time))
    delta_time =  end_time - start_time
    delta_min = round(delta_time.total_seconds() / 60 , 2)
    delta_sec = delta_time.total_seconds()
    print("Time to process:", str(delta_time))
    print("Time to process in min:", str(delta_min))
    print("Time to process in sec:", str(delta_sec))
    print() 

    print()
    print("*** PROGRAM END ***")
    print()