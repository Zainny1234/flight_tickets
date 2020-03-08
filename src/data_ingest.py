import pandas as pd
import logging
import json
import os
from conf.config import DATASET_DIR

logging.basicConfig(filename='test.log', level=logging.DEBUG,
                    format='%(asctime)s:%(funcName)s:%(lineno)d:%(module)s:%(message)s',
                    filemode='w')

logger = logging.getLogger(__name__)


# Load the file

def load_dataset(file_name: str):
    try:
        file_type = os.path.splitext(file_name)[1]
        if file_type == '.xlsx':
            data = pd.read_excel(f'{DATASET_DIR}\{file_name}')
        elif file_type == '.json':
            data = json.loads(open(f'{DATASET_DIR}\{file_name}').read())
        logger.info(f'{file_name} file reading complete')
    except FileNotFoundError as err:
        print('file not found')
        logger.info("File not found", exc_info=True)
        raise
    return data


ml_params = load_dataset('ml_params.json')
x = load_dataset('train.xlsx')
y = load_dataset('ms.json')

if __name__ == '__main__':
    x = load_dataset('train.xlsx')
    y = load_dataset('ms.json')
    x.head(5)
