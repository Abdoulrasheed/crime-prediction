from geopy.geocoders import Nominatim

def preprocess_address(data):
    
    geocoder = Nominatim(user_agent="crime_prediction")
    
    for i, row in enumerate(data):
        
        location = geocoder.geocode(row[6])
        data[i][6] = f"lon: {location.longitude}, lat: {location.latitude}"
        
    return data