from pathlib import Path

def create_directory(dir_out: str, gitkeep: bool = True) -> None:
    """
    Creates the specified directory and an empty .gitkeep file within it.

    Parameters:
        dir_out (str): The path to the directory to be created.
        gitkeep (book): If True, create and empty .gitkeep file.
    Returns:
        None
    """
    # Convert dir_out to a Path object
    directory_path = Path(dir_out)

    # Create the directory, including any necessary intermediate directories
    directory_path.mkdir(parents=True, exist_ok=True)
    print(f"Directory created: {directory_path}")

    # Create the .gitkeep file within the directory
    if gitkeep:
        gitkeep_file = directory_path / ".gitkeep"
        gitkeep_file.touch()
        print(f".gitkeep file created: {gitkeep_file}")

