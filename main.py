import numpy as np
import pandas as pd
from models import Model
from utils import preprocess_address

def get_social_data():
    try:
        return pd.read_csv('datasets/social_sample.csv', keep_default_na=False, na_values=[""])
    except:
        return None

def get_crime_data():
    names = ['fullname', 'phone', 'crime', 'address', 'posts']
    try:
        return pd.read_csv('datasets/crime_main.csv', names=names, keep_default_na=False, na_values=[""])
    except:
        return None

def get_telecom_data():
    try:
        return pd.read_csv('datasets/telecom_main.csv', keep_default_na=False, na_values=[""])
    except:
        return None

def main():
    print("Reading and combining datasets...\n")
    cdata = get_crime_data()
    sdata = get_social_data()
    tdata = get_telecom_data()
    
    combined_dataset = pd.concat([sdata, cdata, tdata], axis=1, join="inner")
    
    model = Model()
    model.train(combined_dataset)
    predicted_result = model.predict()
    print(f"Prediction Results: {predicted_result}")
    model.plot(predicted_result)

if __name__ == "__main__":
    main()