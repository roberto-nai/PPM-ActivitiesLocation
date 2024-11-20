import json
from pathlib import Path
from typing import Union

def extract_data_from_json(json_file: Union[str, Path]):
    """
    Extracts data from a JSON file and returns it as a list of dictionaries.

    Parameters:
        json_file (Union[str, Path]): The JSON file to read. This can be:
                                      - A string representing the file path.
                                      - A Path object from the pathlib library.
                                      
                                      The file is expected to contain a list of dictionaries with the following fields:
                                      - file_name: The name of the file (str).
                                      - case_id_column: The column containing case IDs (str).
                                      - activity_column: The column specifying activities (str).
                                      - timestamp_column: The column containing timestamps (str).
                                      - outcome_column: The column specifying outcomes (str).

    Returns:
        list: A list of dictionaries containing the extracted data, or an empty list if the file is not found or invalid.
    """
    # Convert Path to string if necessary
    if isinstance(json_file, Path):
        json_file = str(json_file)

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{json_file}' does not exist.")
        return []
    except json.JSONDecodeError:
        print(f"Error: The file '{json_file}' is not a valid JSON file.")
        return []
