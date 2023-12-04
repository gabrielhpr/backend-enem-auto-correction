import os
import pandas as pd
from flask import Flask, request, Response
from flask_cors import CORS
from enem_auto_correction.EnemAutoCorrection import EnemAutoCorrection
from tensorflow import keras
from transformers import TFAutoModelForSequenceClassification


# loading model
model = TFAutoModelForSequenceClassification.from_pretrained("gabrielhpr/enem-auto-correction-regression-1")

# checkpoint
checkpoint = "neuralmind/bert-base-portuguese-cased"

# initialize API
app = Flask(__name__)
cors = CORS(app, resources={r"/enem_auto_correction/*": {"origins": "*"}})

@app.route('/enem_auto_correction/predict', methods=['POST'])

def enem_auto_correction_predict():
    test_json = request.get_json()

    # there is data
    if test_json:

        # unique data
        if isinstance(test_json, dict):
            test_raw = pd.DataFrame(test_json, index=[0])

        # multiple data
        else:
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

        # Instantiate Rossmann class
        pipeline = EnemAutoCorrection()

        # data preparation
        df1 = pipeline.data_preparation(test_raw)

        # tokenization
        df2 = pipeline.tokenization(df1, checkpoint)

        # prediction
        df_response = pipeline.get_prediction(model, test_raw, df2)

        return df_response

    # there is no data
    else:
        return Response('{}', status=200, mimetype='application/json')


if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)
