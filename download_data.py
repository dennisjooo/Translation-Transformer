import os
import pandas as pd
import requests
import tarfile
from dotenv import load_dotenv
from config import config
from tqdm import tqdm

load_dotenv()

def download_dataset(target_path: str) -> None:
    """
    Download the English-French translation dataset from statmt.org and extract it to the target path.
    
    Args:
        target_path (str): The destination path where the dataset should be stored
        
    Returns:
        None
    """
    url = "https://www.statmt.org/europarl/v7/fr-en.tgz"
    
    os.makedirs(target_path, exist_ok=True)
    
    print(f"Downloading dataset from {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    temp_file_path = os.path.join(target_path, "temp_download.tgz")
    with open(temp_file_path, 'wb') as temp_file:
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
        for chunk in response.iter_content(chunk_size=block_size):
            size = temp_file.write(chunk)
            progress_bar.update(size)
        progress_bar.close()

    print(f"Extracting files to {target_path}")
    with tarfile.open(temp_file_path, "r:gz") as tar:
        tar.extractall(path=target_path)
    
    os.remove(temp_file_path)
    print(f"Dataset downloaded and extracted to {target_path}")

def merge_data_to_dataframe(data_path: str) -> pd.DataFrame:
    """
    Merge the English and French data into a single DataFrame.
    
    Args:
        data_path (str): Path to the directory containing the extracted files
        
    Returns:
        pd.DataFrame: DataFrame containing 'english' and 'french' columns
    """
    en_file = os.path.join(data_path, "europarl-v7.fr-en.en")
    fr_file = os.path.join(data_path, "europarl-v7.fr-en.fr")
    
    print("Reading files...")
    with open(en_file, 'r', encoding='utf-8') as en_f, open(fr_file, 'r', encoding='utf-8') as fr_f:
        en_lines = en_f.readlines()
        fr_lines = fr_f.readlines()
    
    print("Creating DataFrame...")
    df = pd.DataFrame({
        'en': [line.strip() for line in tqdm(en_lines, desc="Processing English")],
        'fr': [line.strip() for line in tqdm(fr_lines, desc="Processing French")]
    })
    
    data_path = config['data_path']
    print(f"Saving DataFrame to {data_path}...")
    df.to_csv(data_path, index=False)
    
    print(f"DataFrame saved to {data_path}")
    
    # Clean up the .en and .fr files
    print("Cleaning up temporary files...")
    os.remove(en_file)
    os.remove(fr_file)
    
    return df

def ensure_dataset_exists(config: dict) -> pd.DataFrame:
    """
    Check if the dataset exists at the configured path, download it if not found,
    and return the data as a DataFrame.
    
    Args:
        config (dict): Configuration dictionary containing data_folder and data_path
        
    Returns:
        pd.DataFrame: DataFrame containing 'english' and 'french' columns
        
    Raises:
        ValueError: If data_folder or data_path is not specified in the config
    """
    data_folder = config.get('data_folder')
    data_path = config.get('data_path')
    if not data_folder or not data_path:
        raise ValueError("data_folder or data_path not specified in config")
    
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}. Downloading...")
        download_dataset(data_folder)
        print("Merging data into DataFrame...")
        df = merge_data_to_dataframe(data_folder)
    else:
        print(f"Data already exists at {data_path}")
        print("Loading DataFrame...")
        df = pd.read_csv(data_path)
    
    print(f"DataFrame loaded with {len(df)} rows.")
    
    return df

if __name__ == "__main__":
    df = ensure_dataset_exists(config)
    print("\nDataFrame head:")
    print(df.head())
    print(f"\nDataFrame shape: {df.shape}")
