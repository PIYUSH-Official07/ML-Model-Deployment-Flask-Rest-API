import requests
import json
import pandas as pd

# URL of the Flask app API endpoint
url = "http://localhost:5000/predict"

# Load the input JSON data from the input_data.json file
with open('pred_data.json') as f:
    input_data = json.load(f)

# Convert the dictionary to a JSON string (optional, requests will do it automatically)
input_json = json.dumps(input_data)

# Set the headers for the request
headers = {
    "Content-Type": "application/json"
}

# Send POST request to the API with input data
response = requests.post(url, data=input_json, headers=headers)

# Print the response from the Flask app and save it to CSV
if response.status_code == 200:
    prediction = response.json()["prediction"]
    
    # Create a DataFrame from the prediction list
    prediction_df = pd.DataFrame(prediction, columns=["Prediction"])
    
    # Save the predictions to a CSV file
    prediction_df.to_csv("output.csv", index=False)
    
    print("Predictions saved to output.csv")
else:
    print(f"Error: {response.status_code}, {response.text}")




