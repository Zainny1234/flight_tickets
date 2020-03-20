from flask import Flask, request, jsonify
from src.predict import Predict
from src.data_ingest import load_dataset
from conf.config import CATEGORICAL_COLUMNS as cat_cols
import pandas as pd

app = Flask(__name__)

ms = load_dataset('ms.json')['market']
bk_class = load_dataset('class.json')['class']


@app.route('/')
def index():
    return 'this is a Flask'


@app.route('/predict_api', methods=['POST'])
def predict_api():
    print('comes here')
    data = request.get_json(force=True)
    #data = request.json
    # print(data)
    data = pd.DataFrame([data])
    # print(data)
    p = Predict(data)
    price = p.predict(ms, bk_class, cat_cols)
    return {"price": price.to_list()}


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
