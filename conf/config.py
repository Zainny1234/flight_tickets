import os
from pathlib import Path
import pandas as pd

pd.options.display.max_rows = 10
pd.options.display.max_columns = 10

PACKAGE_ROOT = Path().resolve()
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'models'
DATASET_DIR = PACKAGE_ROOT / 'data'

# data
TESTING_DATA_FILE = 'test.xlsx'
TRAINING_DATA_FILE = 'train.xlsx'
TARGET = 'SalePrice'

DAY_OF_BOOKING = '1/3/2019'

CATEGORICAL_COLUMNS = ['Airline', 'Source', 'Destination', 'Additional_Info', 'Date_of_Journey',
                       'Dep_Time', 'Arrival_Time', 'Dep_timeofday', 'Booking_Class',
                       'Arr_timeofday']
