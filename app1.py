from nbformat import write
import streamlit as st
import pickle
import numpy as np
import pandas as pd 
import plotly.graph_objs as go
import matplotlib.pyplot as plt


def main():
    st.title("Child  Mortality Prediction")
    html_temp = """
    <div style="background-color:#c9c04e ;padding:10px">
    <h2 style="color:white;text-align:center;">Child Mortality Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    years = st.text_input("Years","Type Here")
    output=""
    if st.button("Predict"):
        f=pd.read_csv("data2.csv")

        x=np.array(f['Year']).reshape(-1,1)
        y=np.array(f['Estimate']).reshape(-1,1)

        # Getting the square of ID for Polynomial Regression
        from sklearn.preprocessing import PolynomialFeatures 
        poly = PolynomialFeatures(degree=2)
        x = poly.fit_transform(x)
        #print(x)
        from sklearn import linear_model
        model = linear_model.LinearRegression()
        model.fit(x,y)
        poly = PolynomialFeatures(degree=2)
        output=(model.predict(poly.fit_transform([[2020.5+int(years)]])))
    st.success('The Mortality Rate is{}'.format(output))
if __name__=='__main__':
    main()
    v=pd.read_csv('data.csv')
    w=v[['ID','Country','2020(Mortality Rate)']]
    st.write(w)
    country_options=w['Country'].unique().tolist()
    country=st.selectbox('Choose a country',country_options)
    c=w['Country']
    c=list(c)
    print(c)
    inde=(c.index(country))
    st.write(inde)
    m=v['2020(Mortality Rate)']
    st.write("The Mortality Rate For",country," is ",m[inde])

    
    f=pd.read_csv("data2.csv")
    st.line_chart(f['Estimate'])
    
    q=v.iloc[0:5]
    j=f.iloc[20:32]
    st.bar_chart(j['Estimate'])
    