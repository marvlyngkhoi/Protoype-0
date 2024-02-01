import pandas as pd
import numpy as np
from sklearn.linear_model import *
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.metrics import *
import streamlit as st


def label_encode(data_frame,col):
    label_encode = LabelEncoder()
    for c in col:
        data_frame[c] = label_encode.fit_transform(data_frame[c])
    
    return data_frame

def split_data(dataframe,features,labels,t_size = 0.2):
    
    X_train,X_test,y_train,y_test = train_test_split(dataframe[features],dataframe[labels],test_size=t_size,random_state=30)

    return X_train,X_test,y_train,y_test


def train_logistic_reg(featutes,labels):
    model = LogisticRegression()
    model.fit(featutes, labels)  # fit the model to the training set
    
    return model

def log_reg_infer(feature_col,model,container):

    map_dict={
        0:'Setosa',
        1:'Versicolor',
        2:'Virginica'
    }
    f1 = container.number_input(feature_col[0])
    f2 = container.number_input(feature_col[1])
    f3 = container.number_input(feature_col[2])
    f4 = container.number_input(feature_col[3])
    button = container.button('Predict')
    if button:
        pred = model.predict(np.array([[f1,f2,f3,f4]]))
        
        container.write(map_dict[pred[0]])
