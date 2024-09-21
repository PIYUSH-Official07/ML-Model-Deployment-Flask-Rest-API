import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/predict", methods=["POST"])
def predict():
    json = request.json
    query_df = pd.DataFrame(json)

    # Generate predictions
    prediction = model.predict(query_df)
    
    # Save predictions to a CSV file
    output_df = pd.DataFrame(prediction, columns=["Prediction"])
    output_df.to_csv("output.csv", index=False)

    return jsonify({"prediction": list(prediction)})

if __name__ == "__main__":
    flask_app.run(debug=True)
