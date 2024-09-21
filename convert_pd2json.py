import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the csv file
df = pd.read_csv("iris.csv")


# Select independent and dependent variable
X = df[["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]]
# Get the first 50 rows
pred_data = X.head(50)
# Convert the DataFrame to a JSON file
pred_data.reset_index(drop=True, inplace=True)  # Reset index to ensure proper JSON formatting
pred_data.to_json("pred_data.json", orient="records", lines=False,indent=4)