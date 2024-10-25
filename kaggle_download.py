import kagglehub
import os
import shutil
from dotenv import load_dotenv
from config import config

load_dotenv()

def find_csv_file(path: str) -> str:
    """
    Recursively search for a CSV file in the given path and its subdirectories.
    
    Args:
        path (str): Directory path to search for CSV files
        
    Returns:
        str: Relative path to the first CSV file found, or None if no CSV file exists
        
    Example:
        >>> find_csv_file('/path/to/dir')
        'subdir/data.csv'
    """
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path) and item.endswith('.csv'):
            return item
        elif os.path.isdir(item_path):
            result = find_csv_file(item_path)
            if result:
                return os.path.join(os.path.relpath(item_path, path), result)
    return None

def download_dataset(target_path: str) -> None:
    """
    Download the English-French translation dataset from Kaggle and move it to the target path.
    
    Args:
        target_path (str): The destination path where the dataset should be stored
        
    Returns:
        None
    """
    # Download latest version
    download_path = kagglehub.dataset_download(config['kaggle_source_path'])
    print("Path to dataset files:", download_path)

    # Create the target directory if it doesn't exist
    os.makedirs(target_path, exist_ok=True)

    # Try to find the CSV file recursively in the downloaded files
    csv_file = find_csv_file(download_path)
    if not csv_file:
        raise ValueError("CSV file not found in the downloaded dataset")

    # Move the files to the target path
    shutil.move(os.path.join(download_path, csv_file), target_path)
    print("Files moved to:", target_path)

def ensure_dataset_exists(config: dict) -> None:
    """
    Check if the dataset exists at the configured path and download it if not found.
    
    Args:
        config (dict): Configuration dictionary containing data_path
        
    Returns:
        None
        
    Raises:
        ValueError: If data_path is not specified in the config
    """
    data_path = config.get('data_path')
    if not data_path:
        raise ValueError("data_path not specified in config")
    
    # Check if data already exists at the specified path
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}. Downloading...")
                
        # Download the dataset
        download_dataset(data_path.split('/')[0])
    else:
        print(f"Data already exists at {data_path}")


if __name__ == "__main__":
    ensure_dataset_exists(config)
    
    