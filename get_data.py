import pathlib
import zipfile

import requests

DATA_FOLDER = pathlib.Path('data')
DATA_FOLDER.mkdir(exist_ok=True)
ZIP_FILE = pathlib.Path(DATA_FOLDER / 'ml-1m.zip')

# Get the 1M dataset because it contains demographic data
data = requests.get('https://files.grouplens.org/datasets/movielens/ml-1m.zip')

with open(ZIP_FILE, 'wb') as f:
    f.write(data.content)

with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
    zip_ref.extractall(DATA_FOLDER)

ZIP_FILE.unlink()
