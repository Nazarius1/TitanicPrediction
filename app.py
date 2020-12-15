# Dependencies
import sys
import traceback
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
# import numpy as np
# import pickle

app = Flask(__name__)
# model = joblib.load('model.pkl')

@app.route("/", methods=['GET'])
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if model:
        try:
            # int_features = [int(x) for x in request.form.values()]
            input = [{"Age": request.form['age'],"Sex":request.form['sex'],"Embarked":request.form['embarked']}]
            # final_features = [np.array(int_features)]
            query =  pd.get_dummies(pd.DataFrame(input))
            query = query.reindex(columns=model_columns, fill_value=0)
            prediction = model.predict(query)
            if prediction == 0:
                str_prediction='No'
            else:
                str_prediction='Yes'

            return render_template('index.html',  prediction_text='The passenger would survived? ' + str_prediction)

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        prediction_text = 'Train the model first, no model here to use'



if __name__ == '__main__':
    try:
        port = int(sys.argv[1])  # This is for a command-line input

    except:
        port = 12345  # If you don't provide any port the port will be set to 12345
    model = joblib.load('model.pkl')
    model_columns = joblib.load("model_columns.pkl")  # Load "model_columns.pkl"

    app.run(port=port, debug=True)
