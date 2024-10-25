import kagglehub
import os
import shutil
from dotenv import load_dotenv

load_dotenv()

# Download latest version
path = kagglehub.dataset_download("dhruvildave/en-fr-translation-dataset")
print("Path to dataset files:", path)

# Set this file's directory as the working directory
cwd = os.path.dirname(os.path.abspath(__file__))

# Create the data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Move the files to the CWD
shutil.move(path, os.path.join(cwd, 'data'))

print("Files moved to:", os.path.join(cwd, 'data'))