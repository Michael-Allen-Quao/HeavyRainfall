import streamlit as st
import sklearn
from joblib import load

# Loading the trained model and scaler
model = load('random_forest_model.joblib')
scaler = load('scaler.joblib')

# Function to preprocess input data
def preprocess_input(data):
    scaled_data = scaler.transform(data)
    return scaled_data

# Function to make a prediction using the model
def predict(data):
    preprocessed_data = preprocess_input(data)
    predictions = model.predict(preprocessed_data)
    return predictions

# Defining the main function to run the app
def main():
    st.title('Heavy Rainfall Prediction Interface')
    st.write('Enter the weather data below to make a prediction.')

    # Collecting the users input
    Temperature = st.number_input('Temperature (C)', value=0.0)
    Dew_Point_Temperature = st.number_input('Dew Point Temperature (C)', value=0.0)
    Humidity = st.number_input('Humidity (%)', value=0.0)
    Visibility = st.number_input('Visibility (km)', value=0.0)

    # Creating a feature array from the users input
    user_data = [[Temperature, Dew_Point_Temperature, Humidity, Visibility]]

    # Making a prediction
    if st.button('Predict'):
        prediction = predict(user_data)
        weather_labels = {0: 'Heavy Rain Unlikely', 1: 'Heavy Rain Unlikely', 2: 'Heavy Rain Unlikely', 3: 'Heavy Rain Unlikely', 4: 'Heavy Rain Unlikely', 5: 'Heavy Rain Likely', 6: 'Heavy Rain Unlikely', 7: 'Heavy Rain Unlikely'}
        predicted_label = weather_labels[prediction[0]]
        st.write(f'Predicted weather condition: {predicted_label}')

if __name__ == '__main__':
    main()