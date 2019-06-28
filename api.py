# Dependencies
from flask import Flask, request, jsonify
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import json

# Your API definition
app = Flask(__name__)

@app.route("/")
def hello():
    return "Welcome to machine learning model APIs!"


@app.route('/predict', methods=['POST'])
def predict():
    if grb:
        try:
            json_ = request.json
            print(str(json_))
            json_['make'] = vehicle_make[str(json_['make']).lower()]
            json_['model'] = vehicle_model[str(json_['model']).lower()]
            json_['transmissionType'] = vehicle_transmissionType[str(json_['transmissionType']).lower()]
            json_['vehicleStyle'] = vehicle_vehicleStyle[str(json_['vehicleStyle']).lower()]
            
            print(json_)
            query = pd.DataFrame(json_, index=[0])
            #query = query.reindex(columns=model_columns, fill_value=0)
            print(query)
            prediction = list(grb.predict(query))

            return jsonify({'status': '200','price': prediction[0]})

        except:

            return jsonify({'status': '500', 'price': NULL})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    """
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345
    """
    grb = joblib.load("price_pred.pkl") # Load "model.pkl"
    print ('price_pred loaded')
    
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')
    
    vehicle_make = joblib.load('vehicle_make.pkl')
    print ('vehicle_make loaded')
    #print (vehicle_make)
    
    vehicle_model = joblib.load('vehicle_model.pkl')
    print ('vehicle_model loaded')
    #print (vehicle_model)
    
    vehicle_transmissionType = joblib.load('vehicle_transmissionType.pkl')
    print ('vehicle_transmissionType loaded')
    #print (vehicle_transmissionType)
    
    vehicle_vehicleStyle = joblib.load('vehicle_vehicleStyle.pkl')
    print ('vehicle_vehicleStyle loaded')
    #print (vehicle_vehicleStyle)
    
    app.run(port=5400, debug=False)