from src.final_preprocess import PreProcess
import joblib
import os


class Predict(PreProcess):
    def __init__(self, X):
        super().__init__(X)
        #self.x = x

    def predict(self, ms, bk_class, cat_cols):
        self.x = self.preprocess(ms, bk_class, cat_cols, False)
        model = joblib.load(os.path.join(os.getcwd(), 'models', 'lgbm_es.sav'))
        self.x['Price'] = model.predict(self.x)
        return self.x


if __name__ == "__main__":
    from src.data_ingest import load_dataset
    from conf.config import CATEGORICAL_COLUMNS as cat_cols
    x = load_dataset('test.xlsx')
    #x.columns = x.columns.str.replace(":", "_")
    ms = load_dataset('ms.json')['market']
    bk_class = load_dataset('class.json')['class']

    p = Predict(x)
    data = p.predict(ms, bk_class, cat_cols)
