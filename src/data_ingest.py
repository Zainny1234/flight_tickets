import pandas as pd
import logging
import json
import os
from conf.config import DATASET_DIR

_logger = logging.getLogger(__name__)


# Load the file

def load_dataset(file_name: str):
    file_type = os.path.splitext(file_name)[1]
    if file_type == '.xlsx':
        #print(f'{DATASET_DIR}')
        data = pd.read_excel(f'{DATASET_DIR}\{file_name}')
    elif file_type == '.json':
        data = json.loads(open(f'{DATASET_DIR}\{file_name}').read())
    else:
        raise Exception(f'{file_type} is not supported')
    return data


ml_params = load_dataset('ml_params.json')
x = load_dataset('train.xlsx')
y = load_dataset('ms.json')


if __name__ == '__main__':
    x = load_dataset('train.xlsx')
    y = load_dataset('ms.json')
    x.head(5)
