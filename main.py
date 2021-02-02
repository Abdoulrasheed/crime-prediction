import pandas as pd
from pandas import DataFrame
from models import Model
from utils import preprocess_address

def get_social_data():
    return pd.read_csv('datasets/social_sample.csv')

def get_crime_data():
    names = ['surname', 'first_name', 'phone_number','Age', 'crime', 'date_of_crime', 'location']
    return pd.read_csv('datasets/crime_main.csv', names=names)

def get_telecom_data():
    return pd.read_csv('datasets/telecom_main.csv')

def main():
    sdata = get_social_data()
    cdata = get_crime_data()
    #tdata = get_telecom_data()
    
    datasets = {"social": sdata, "crime": cdata, "telecom": tdata}
    model = Model()
    model.crime_data = cdata
    model.social_data = sdata
    model.telecom_data = tdata
    
    model.train()
    output = model.predict()
    
    #draw matplotlib
    
    #processed_data = preprocess_address(cdata.values)
    
    #print(processed_data)

if __name__ == "__main__":
    main()