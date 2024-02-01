import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def read_csv_file(file_path):
    if file_path:
        file = pd.read_csv(file_path)
        columns = list(file.columns)
        return file,columns

    return None,[]

@st.cache_data
def get_data(data_frame,opt):
    if not data_frame.empty and opt:        
        if opt[0]!='All':
            
            return data_frame[opt]
        else:
            return data_frame

# def get_tabs(tabs_id):
#     tab1, tab2, tab3 ,tab4 = tabs_id.tabs(["Display Data", "Visualize", "Edit Data","ML Toolkit"])


     
