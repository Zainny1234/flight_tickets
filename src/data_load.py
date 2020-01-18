import pandas as pd
from flight_tickets.config import config
import logging



_logger = logging.getLogger(__name__)


def load_dataset(*, file_name: str
                 ) -> pd.DataFrame:
    _data = pd.read_csv(f'{config.DATASET_DIR}/{file_name}')
    return _data