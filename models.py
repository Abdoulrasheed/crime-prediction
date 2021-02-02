import tensorflow as tf

class Model:
    crime_data = None
    social_data = None
    telecom_data = None
    
    def __init__:
        pass
    
    def train(self, datasets):
        model = tf.keras.Sequential([tf.keras.layers.Dense(units=1,input_shape=[1])])
        model.compile(optimizer='sgd',loss='mean_squared_error')

        #Training Neural Net
        model.fit(self.crime_data, self.social_data, self.telecom_data,epochs = 5000)
    
    def predict(self):
        result = model.predict([10.0])
        return result