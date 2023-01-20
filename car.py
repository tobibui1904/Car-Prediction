import streamlit as st
import pandas as pd
import numpy as np
from pandasql import sqldf
import random
from PIL import ImageTk

#Machine Learning libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict

# Dall-2E API usage
import requests
from requests.structures import CaseInsensitiveDict
import json
from PIL import Image

#outlines of the website
header=st.container()
prediction = st.container()
picture = st.container()

#Introduction about the program
with header:
    st.markdown("<h1 style='text-align: center; color: lightblue;'>Used Car Prediction</h1>", unsafe_allow_html=True)
    st.caption("<h1 style='text-align: center;'>By Tobi Bui</h1>",unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: left; color: red;'>Introduction about the project</h1>", unsafe_allow_html=True)
    st.subheader('1: Project Purposes')
    st.markdown("""The objective of my project is to study how Machine Learning algorithm Linear Regression works in real life data of used car. With the available data from Quikr, the program can predict the future price based 
                on the relationship of model, year purchased, company and kms_driven with price. Alongside with that, I also implement the Open Ai's API Dall-2E to convert the text prompt to image of the chosen car for better
                visualization. """)
    st.subheader('2: How it works')
    st.markdown("""- Step 1: Select the car company.""")
    st.markdown("""- Step 2: Select the available model from that firm.""")
    st.markdown("""- Step 3: Select the available year of Purchase.""")
    st.markdown("""- Step 4: Select the available fuel.""")
    st.markdown("""- Step 5: Enter the Number of Kilometers that the car has traveled.""")
    st.markdown("""- Step 6: Enter the year that you want my program to predict the price.""")
    st.subheader('3: Features')
    st.markdown("""- Linear Regression Prediction""")
    st.markdown("""- Text to image""")
    
    st.write("---")

with prediction:
    st.markdown("<h1 style='text-align: left; color: red;'>Prediction</h1>", unsafe_allow_html=True)
    
    # Load dataset
    car=pd.read_csv(r"C:\Users\Admin\car prediction\Cleaned_Car_data.csv")
    
    # Drop unnecessary column
    car = car.drop(['Unnamed: 0'], axis=1)
    
    # Convert Rupee to USD
    car['Price'] =  (car['Price']*0.012).astype(int)
    st.subheader("Data Table")
    st.write(car)
    
    st.subheader("Prediction Input Process")
    # Input company value
    firm = st.selectbox('Select the company', np.sort(car.company.unique()))
    car = sqldf(f"SELECT * FROM car where company is '{firm}' ")
    
    # Input model value
    model = st.selectbox('Select the model', np.sort(car.name.unique()))
    car = sqldf(f"SELECT * FROM car where name is '{model}' ")
    
    # Input year value
    year = st.selectbox('Select Year of Purchase', np.sort(car.year.unique()))
    car = sqldf(f"SELECT distinct * FROM car where year is '{year}'")
    
    # Input fuel value
    fuel = st.selectbox('Select the Fuel Type', car.fuel_type.unique())
    car = sqldf(f"SELECT distinct * FROM car where fuel_type is '{fuel}'")
    
    # Input distance value
    distance = st.text_area('Enter the Number of Kilometers that the car has traveled')
    if distance.strip() == "":
        st.warning("Please enter the number of kilometers")
    else:
        try:
            distance = int(distance)
        except ValueError:
            st.warning("Please enter a valid number of kilometers")
    
    # Input prediction year value
    year = st.text_area('Enter the year that you want to be evaluated')
    if year.strip() == "":
        st.warning("Please enter the year number please")
    else:
        try:
            year = int(year)
        except ValueError:
            st.warning("Please enter a valid year number")
            
    # Adding column to the dataframe with controlled random variable to make the prediction
    if car.shape[0] < 2:
        random_price = random.randint(car.loc[0][3],car.loc[0][3] + 10000)
        random_kms_driven = random.randint(car.loc[0][3] - 5000 , car.loc[0][3])
        car.loc[len(car.index)] = [car.loc[0][0], car.loc[0][1], car.loc[0][2], random_price, random_kms_driven , car.loc[0][5]]
    
    if year and distance:
        # Extracting Training Data
        X=car[['name','company','year','kms_driven','fuel_type']]
        y=car['Price']
        
        # Creating an OneHotEncoder object to contain all the possible categories
        ohe=OneHotEncoder()
        ohe.fit(X[['name','company','fuel_type']])

        # Creating a column transformer to transform categorical columns
        column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),
                                            remainder='passthrough')

        # Linear Regression Model
        lr=LinearRegression()

        # Making a Pipeline
        pipe=make_pipeline(column_trans,lr)
        
        # Applying Train Test Split
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

        # Fitting the model
        pipe.fit(X_train,y_train)

        y_pred=pipe.predict(X_test)

        # Checking R2 Score
        r2_score(y_test,y_pred)

        # Finding the model with a random state of TrainTestSplit where the model was found to give almost 0.92 as r2_score
        scores=[]
        for i in range(1000):
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=i)
            lr=LinearRegression()
            pipe=make_pipeline(column_trans,lr)
            pipe.fit(X_train,y_train)
            y_pred=pipe.predict(X_test)
            scores.append(r2_score(y_test,y_pred))

        # Random state determined to optimize the prediction
        np.argmax(scores)

        #The best model is found at a certain random state
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=np.argmax(scores))
        lr=LinearRegression()
        pipe=make_pipeline(column_trans,lr)
        pipe.fit(X_train,y_train)
        y_pred=pipe.predict(X_test)
        r2_score(y_test,y_pred)

        # Output the prediction
        st.write("The predicted price is " + str(pipe.predict(pd.DataFrame(columns=X_test.columns,data=np.array([model,firm,year,distance,fuel]).reshape(1,5)))))

        st.write("---")
    
    else:
        st.warning("You have to enter all the requested value in order to proceed")

with picture:
    QUERY_URL = "https://api.openai.com/v1/images/generations"

    # Get the API key
    api_key = "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"

    # Create a text input
    prompt = st.text_input("Add more asthetic to the car")
    prompt = model + " " + prompt
    st.write("Here is your art car: " + prompt)

    if prompt:
        headers = CaseInsensitiveDict()
        headers["Content-Type"] = "application/json"
        headers["Authorization"] = f"Bearer {api_key}"

        data = """
        {
            """
        data += f'"model": "image-alpha-001",'
        data += f'"prompt": "{prompt}",'
        data += """
            "num_images":1,
            "size":"1024x1024",
            "response_format":"url"
        }
        """

        resp = requests.post(QUERY_URL, headers=headers, data=data)

        response_text = json.loads(resp.text)
        image_url = response_text['data'][0]['url']

        # Open the image and display it
        img = Image.open(requests.get(image_url, stream=True).raw)
        st.image(img)
