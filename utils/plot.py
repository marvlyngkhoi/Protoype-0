import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_corr(dataframe,_tab):
    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(dataframe.corr(),cmap='coolwarm',fmt=".2f")
    _tab.pyplot(fig)


def plot_hist(dataframe,_tab):
    fig = plt.figure(figsize=(8, 5))
    sns.histplot(dataframe,kde='True')
    _tab.pyplot(fig)



def plot_pair(dataframe,_tab):
    fig = plt.figure(figsize=(8, 5))
    sns.histplot(dataframe,kde='True')
    _tab.pyplot(fig)



def plot_box(dataframe,_tab):
    fig = plt.figure(figsize=(8, 5))
    sns.boxplot(x=dataframe)
    _tab.pyplot(fig)

