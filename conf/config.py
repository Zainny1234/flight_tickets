import os
from pathlib import Path
import pandas as pd


pd.options.display.max_rows = 10
pd.options.display.max_columns = 10


PACKAGE_ROOT = Path().resolve()
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'models'
DATASET_DIR = PACKAGE_ROOT / 'data'

# data
TESTING_DATA_FILE = 'test.csv'
TRAINING_DATA_FILE = 'train.csv'
TARGET = 'SalePrice'

