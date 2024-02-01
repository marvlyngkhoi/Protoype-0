import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from utils.plot import *
from utils.display import *
from utils.process import *
from utils.ml import *

st.set_page_config(layout="wide")


css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1.5rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] { margin-right: 40px; }
</style>
'''

st.markdown(css, unsafe_allow_html=True)



def main():
    dummy_file = st.sidebar.checkbox('Use Dummy Datset')

    uploaded_csv = st.sidebar.file_uploader('Choose a CSV file...')
    
    

    if dummy_file and not uploaded_csv:
        uploaded_csv= "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"
        
    data_frame,cols =  read_csv_file(uploaded_csv)

    
    copy_df = data_frame.copy() if uploaded_csv else pd.DataFrame()

    if 'copy_df' not in st.session_state and not copy_df.empty:
        st.session_state.copy_df = copy_df
        
    
    cols.append('All')
    
    if len(cols)>0:
        
        option = st.multiselect('Select Columns',cols,default='All')
    
    tab1, tab2, tab3 ,tab4 = st.tabs(["Display Data", "Visualize", "Process Data","ML Toolkit"])


    with tab1:
        col1,col2,col3,col4 = st.columns(4)

        disp = col1.toggle('Show Data')
        null = col2.toggle('Count Null')
        desc = col3.toggle("Description")
        info = col4.toggle('Info')
        
        
        tab1_col1,tab1_col2 = st.columns([0.6,0.4])
        if uploaded_csv and len(option)>0:                      
            
            if disp :
               tab1_col1.write('First 5 rows')
               tab1_col1.dataframe(get_data(data_frame,option).head()) 
            if desc:
                tab1_col1.write('Stats and  Description')

                numerical = get_data(data_frame,option).select_dtypes(include='number')
                non_numerical = get_data(data_frame,option).select_dtypes(include='object')
                if not numerical.empty:
                    tab1_col1.write(numerical.describe().T) 
                
                if not non_numerical.empty:
                    tab1_col1.write(non_numerical.describe().T) 


            if info:
                tab1_col2.write('Info')
                tab1_col2.write(get_data(data_frame,option).dtypes)

                tab1_col2.write('Number of Unique Instance of each non-numerical Columns')
                tab1_col2.write(get_data(data_frame,option).select_dtypes(include='object').nunique().to_dict())


            if null:
                tab1_col2.write('Number of Null values for the selected columns')
                tab1_col2.write(get_data(data_frame,option).isnull().sum().to_dict()) 

            
    with tab2:

        plot_option = tab2.selectbox('Plot Type',
            ['Correlation Plot',
             'Distribution Plot',
             'Pair Plot',
             'Box Plot'
             ]
        )
        tab2_col1,tab2_col2 = st.columns([0.3,0.7])
        
        if plot_option == "Correlation Plot":
            if 'All' not in option:
                corr_var = tab2.multiselect('Select Features',option)
            else:
                corr_var = tab2.multiselect('Select Features',cols)
            
            tab2_button = tab2.button('Plot')
            if tab2_button:
                num_data = get_data(data_frame,corr_var).select_dtypes(include='number')
                
                plot_corr(num_data,tab2)
        
        elif plot_option=='Distribution Plot':
            if 'All' not in option:
                dist_var = tab2_col1.multiselect('Select Features',option)
            else:
                dist_var = tab2_col1.multiselect('Select Features',cols)
            
            tab2_button = tab2_col1.button('Plot')
            if tab2_button:
                for col in dist_var:
                    plot_data = get_data(data_frame,col)
                    plot_hist(plot_data,tab2_col2)

        elif plot_option=='Box Plot':
            if 'All' not in option:
                box_var = tab2_col1.multiselect('Select Features',option)
            else:
                box_var = tab2_col1.multiselect('Select Features',cols)
            
            tab2_button = tab2_col1.button('Plot')
            
            if tab2_button:
                for col in box_var:
                    plot_data = get_data(data_frame,col)
                    plot_box(plot_data,tab2_col2)
                
                
            
        
    with tab3:
        
        edit_option = tab3.selectbox(label='Edit',
                options=['Delete Na','Impute','Normalize','Encode'],
                label_visibility='hidden'                             
        )


        tab3_col1,tab3_col2 = tab3.columns([0.4,0.6])
        
        if edit_option=='Delete Na':            
            
            tab3_button = tab3_col1.button('Delete Null Values')
            if tab3_button:
                st.session_state.copy_df = st.session_state.copy_df.dropna(axis = 0)
                tab3_col2.write(st.session_state.copy_df.head())
                tab3_col2.write(st.session_state.copy_df.isnull().sum().to_dict())
            
        if edit_option =='Encode':

            if 'All' not in option:
                encode_var = tab3_col1.multiselect('Select Features',option)
            else:
                encode_var = tab3_col1.multiselect('Select Features',cols,key = 'Cols')

            tab3_button = tab3_col1.button('Encode')
            if tab3_button:
                st.session_state.copy_df = label_encode(st.session_state.copy_df,encode_var)
                tab3_col2.write(st.session_state.copy_df.head())
                
            
    with tab4:
        tab4_container = tab4.container()
        tab4_row1_col1,tab4_row1_col2 = tab4_container.columns(2)

        if 'All' not in option:
            feature_var = tab4_row1_col1.muliselect('Select Features',option)
        else:
            feature_var = tab4_row1_col1.multiselect('Select Features',cols,key = 'tab4_con_col1')
        if 'All' not in option:
            label_var = tab4_row1_col2.muliselect('Select label',option)
        else:
            label_var = tab4_row1_col2.multiselect('Select label',cols,key = 'tab4_con_col2')
        
        # tab4.write(label_var)
        # tab4.write(feature_var)
        tab4_col1,tab4_col2  = tab4.columns([0.4,0.6])

        moedel_option = tab4_col1.selectbox(
            "Choose Model",
            options=['Logistic Regression'],
            
        )
        tab4_button = tab4_col1.button('Train Model')
        if not copy_df.empty:
            X_train,X_test,y_train,y_test = split_data(st.session_state.copy_df,feature_var,label_var)

        if tab4_button:
            if moedel_option=='Logistic Regression':
                model = train_logistic_reg(X_train,y_train)
                tab4_col2.write('Model Training Done')
                tab4_col2.write(f'Model Accuracy {model.score(X_test,y_test)}')

                if 'log_reg' not in st.session_state:
                    st.session_state.log_reg = model


        tab4_tog = tab4_col1.toggle('Inference')
        if tab4_tog and 'log_reg' in st.session_state:
            
            log_reg_infer(feature_var,st.session_state.log_reg,tab4_col2)
        




main()
