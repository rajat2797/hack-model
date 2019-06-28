#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:13:21 2019

@author: aspira.tripathi
"""
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.externals import joblib


car_info = pd.read_csv("data.csv",delimiter = ",",verbose= True, usecols = [0,1,2,5,6,11,15])
car_info.rename(columns = {"MSRP":"Price", "Engine Cylinders":"EngineCylinders","Transmission Type":"TransmissionType","Vehicle Style":"VehicleStyle"}, inplace = True)
cinfo1 = pd.read_csv("cars_data.csv", delimiter = ",", verbose = True, usecols = [9,8,7,6,4,3,0])
cinfo1.rename(columns = {"year":'Year',"price":"Price","make":"Make", "model":'Model',"engine_cylinders":"EngineCylinders","vehicle_transmission":"TransmissionType","vehicle_style":"VehicleStyle"}, inplace = True)
#cinfo2 = pd.read_csv("cars_big.json", delimiter=",", verbose =True)
cinfo = pd.concat([car_info, cinfo1],ignore_index=True)

#cinfo.loc[ cinfo['Transmission Type'].str.contains('Auto', case=False), 'Transmission Type' ] = 'Automatic'

cinfo['EngineCylinders'].replace(' ', np.nan, inplace=True)
cinfo['EngineCylinders'].replace('Hybrid', np.nan, inplace=True)

cinfo.dropna(subset =['EngineCylinders', 'VehicleStyle','TransmissionType'], inplace = True)

cinfo['TransmissionType'] = cinfo.TransmissionType.apply(lambda x: 'Automatic' if 'Auto' in x  else x)
cinfo['TransmissionType'] = cinfo.TransmissionType.apply(lambda x: 'Automatic' if 'AUTO' in x  else x)
cinfo['TransmissionType'] = cinfo.TransmissionType.apply(lambda x: 'Manual' if 'MAN' in x  else x)
cinfo['TransmissionType'] = cinfo.TransmissionType.apply(lambda x: 'Manual' if 'Man' in x  else x)
cinfo['TransmissionType'] = cinfo.TransmissionType.apply(lambda x: 'Manual' if 'UNKNOWN' in x  else x)
cinfo['TransmissionType'] = cinfo.TransmissionType.apply(lambda x: 'Manual' if 'Single-Speed' in x  else x)
cinfo['TransmissionType'] = cinfo.TransmissionType.apply(lambda x: 'Manual' if " " in x  else x)
cinfo['TransmissionType'] = cinfo.TransmissionType.apply(lambda x: 'Manual' if 'DIRECT_DRIVE' in x  else x)

cinfo['VehicleStyle'] = cinfo.VehicleStyle.apply(lambda x: 'Coupe' if 'Coupe' in x  else x)
cinfo['VehicleStyle'] = cinfo.VehicleStyle.apply(lambda x: 'SUV' if 'SUV' in x  else x)
cinfo['VehicleStyle'] = cinfo.VehicleStyle.apply(lambda x: 'Sedan' if 'Sedan' in x  else x)
cinfo['VehicleStyle'] = cinfo.VehicleStyle.apply(lambda x: 'Wagon' if 'Wagon' in x  else x)
cinfo['VehicleStyle'] = cinfo.VehicleStyle.apply(lambda x: 'Hatchback' if 'Hatchback' in x  else x)
cinfo['VehicleStyle'] = cinfo.VehicleStyle.apply(lambda x: 'Minivan' if 'Minivan' in x  else x)
cinfo['VehicleStyle'] = cinfo.VehicleStyle.apply(lambda x: 'CabPickup' if 'Pickup' in x  else x)
cinfo['VehicleStyle'] = cinfo.VehicleStyle.apply(lambda x: 'Van' if 'Van' in x  else x)
cinfo['VehicleStyle'] = cinfo.VehicleStyle.apply(lambda x: 'Truck' if 'Truck' in x  else x)
cinfo['VehicleStyle'] = cinfo.VehicleStyle.apply(lambda x: 'SportUtility' if 'Utility' in x  else x)
cinfo['VehicleStyle'] = cinfo.VehicleStyle.apply(lambda x: 'Coupe' if 'Coupe' in x  else x)
     
cinfo['EngineCylinders'] = cinfo.EngineCylinders.apply(lambda x: 6.0 if isinstance(x, str) and '6' in x  else x)
cinfo['EngineCylinders'] = cinfo.EngineCylinders.apply(lambda x: 8.0 if isinstance(x, str) and '8' in x  else x)
cinfo['EngineCylinders'] = cinfo.EngineCylinders.apply(lambda x: 12.0 if isinstance(x, str) and '12' in x  else x)
cinfo['EngineCylinders'] = cinfo.EngineCylinders.apply(lambda x: 0.0 if isinstance(x, str) and '0' in x  else x)
cinfo['EngineCylinders'] = cinfo.EngineCylinders.apply(lambda x: 5.0 if isinstance(x, str) and '5' in x  else x)
cinfo['EngineCylinders'] = cinfo.EngineCylinders.apply(lambda x: 4.0 if isinstance(x, str) and '4' in x  else x)
cinfo['EngineCylinders'] = cinfo.EngineCylinders.apply(lambda x: 0.0 if isinstance(x, str) and 'Electric' in x  else x)
cinfo['EngineCylinders'] = cinfo.EngineCylinders.apply(lambda x: 10.0 if isinstance(x, str) and '10' in x  else x)
cinfo['EngineCylinders'] = cinfo.EngineCylinders.apply(lambda x: 16.0 if isinstance(x, str) and '16' in x  else x)
cinfo['EngineCylinders'] = cinfo.EngineCylinders.apply(lambda x: 3.0 if isinstance(x, str) and '3' in x  else x)

cinfo['Make'] = cinfo['Make'].apply(lambda x: x.lower())
cinfo['Model'] = cinfo['Model'].apply(lambda x: x.lower())
cinfo['TransmissionType'] = cinfo['TransmissionType'].apply(lambda x: x.lower())
cinfo['VehicleStyle'] = cinfo['VehicleStyle'].apply(lambda x: x.lower())

neworder= [1,2,6,0,4,5,3];
cinfo = cinfo[cinfo.columns[neworder]]

car_info_X = cinfo.values[:,0:-1]
car_info_Y = cinfo.values[:,-1]

labelencoder = LabelEncoder()

car_info_X[:,0] = labelencoder.fit_transform(car_info_X[:,0])

keys = labelencoder.classes_
values = labelencoder.transform(labelencoder.classes_)
dictionary = dict(zip(keys, values))
joblib.dump(dictionary, 'vehicle_make.pkl', protocol=2)

car_info_X[:,1] = labelencoder.fit_transform(car_info_X[:,1])

keys = labelencoder.classes_
values = labelencoder.transform(labelencoder.classes_)
dictionary = dict(zip(keys, values))
joblib.dump(dictionary, 'vehicle_model.pkl', protocol=2)

car_info_X[:,4] = labelencoder.fit_transform(car_info_X[:,4])

keys = labelencoder.classes_
values = labelencoder.transform(labelencoder.classes_)
dictionary = dict(zip(keys, values))
joblib.dump(dictionary, 'vehicle_transmissionType.pkl', protocol=2)

car_info_X[:,5] = labelencoder.fit_transform(car_info_X[:,5])

keys = labelencoder.classes_
values = labelencoder.transform(labelencoder.classes_)
dictionary = dict(zip(keys, values))
joblib.dump(dictionary, 'vehicle_vehicleStyle.pkl', protocol=2)

"""
onehotencoder = OneHotEncoder(categorical_features = [0,1,5])
car_info_X = onehotencoder.fit_transform(car_info_X).toarray()
"""

X_train, X_test, Y_train, Y_test = train_test_split(car_info_X, car_info_Y, test_size=0.20)

gbr = GradientBoostingRegressor(loss ='ls', max_depth=6)
gbr.fit(X_train, Y_train)
predictions = gbr.predict(X_test)
residual = Y_test-predictions
acc = r2_score(Y_test,predictions)
joblib.dump(gbr, 'price_pred.pkl', protocol=2)
model_columns = list(cinfo.columns[:-1])
joblib.dump(model_columns, 'model_columns.pkl', protocol=2)