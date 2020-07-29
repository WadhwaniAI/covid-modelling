import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyxlsb import convert_date

# Returns the age bracket given the age of the individual. Currently, divided into buckets of 10 years
def get_age_bracket(row,age_column='Age'):
    age_low = int(row[age_column]/10)*10
    age_high = age_low+9
    return str(age_low)+"-"+str(age_high)
    
# Returns the age and gender distribution (time - series) for the input dataframe,df.
def age_gender_distribution(df,date_col='Date',age_col='Age Bracket',gen_cal='Gender'):
    x1 = df.groupby([date_col,age_col]).agg('size').reset_index().rename(columns={0:'patient count'})
    x2 = df.groupby([date_col,gen_cal]).agg('size').reset_index().rename(columns={0:'patient count'})
    return x1,x2

def plot_multiple_time_series(df,column_divide,column_x,column_y,xlabel,ylabel,title,filename):
    output_dir = "Plots2/"
    column_vals = df[column_divide].unique()
    column_vals.sort()
    fig = plt.figure(figsize=(25,14))
    for val in column_vals:
        time_series = df[df[column_divide]==val]
        plt.plot(time_series[column_x],time_series[column_y],label=val,linewidth=3)
    fig.autofmt_xdate()
    plt.legend(prop={'size': 20})
    plt.ylabel(ylabel,fontsize=20)
    plt.xlabel(xlabel,fontsize=20)
    plt.title(title,fontsize=30)
    plt.savefig(output_dir+filename)
