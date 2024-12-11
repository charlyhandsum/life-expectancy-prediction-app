import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load('life_expectancy_model.pkl')

# List of countries (you can modify this list based on your dataset)
countries = ['Afghanistan', 'Albania', 'Algeria', 'Angola',
       'Antigua and Barbuda', 'Argentina', 'Armenia', 'Australia',
       'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh',
       'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan',
       'Bolivia (Plurinational State of)', 'Bosnia and Herzegovina',
       'Botswana', 'Brazil', 'Brunei Darussalam', 'Bulgaria',
       'Burkina Faso', 'Burundi', "Côte d'Ivoire", 'Cabo Verde',
       'Cambodia', 'Cameroon', 'Canada', 'Central African Republic',
       'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Congo',
       'Cook Islands', 'Costa Rica', 'Croatia', 'Cuba', 'Cyprus',
       'Czechia', "Democratic People's Republic of Korea",
       'Democratic Republic of the Congo', 'Denmark', 'Djibouti',
       'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt',
       'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia',
       'Ethiopia', 'Fiji', 'Finland', 'France', 'Gabon', 'Gambia',
       'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 'Guatemala',
       'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras',
       'Hungary', 'Iceland', 'India', 'Indonesia',
       'Iran (Islamic Republic of)', 'Iraq', 'Ireland', 'Israel', 'Italy',
       'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati',
       'Kuwait', 'Kyrgyzstan', "Lao People's Democratic Republic",
       'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Lithuania',
       'Luxembourg', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives',
       'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius',
       'Mexico', 'Micronesia (Federated States of)', 'Monaco', 'Mongolia',
       'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia',
       'Nauru', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua',
       'Niger', 'Nigeria', 'Niue', 'Norway', 'Oman', 'Pakistan', 'Palau',
       'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines',
       'Poland', 'Portugal', 'Qatar', 'Republic of Korea',
       'Republic of Moldova', 'Romania', 'Russian Federation', 'Rwanda',
       'Saint Kitts and Nevis', 'Saint Lucia',
       'Saint Vincent and the Grenadines', 'Samoa', 'San Marino',
       'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia',
       'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia',
       'Solomon Islands', 'Somalia', 'South Africa', 'South Sudan',
       'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Swaziland', 'Sweden',
       'Switzerland', 'Syrian Arab Republic', 'Tajikistan', 'Thailand',
       'The former Yugoslav republic of Macedonia', 'Timor-Leste', 'Togo',
       'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey',
       'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine',
       'United Arab Emirates',
       'United Kingdom of Great Britain and Northern Ireland',
       'United Republic of Tanzania', 'United States of America',
       'Uruguay', 'Uzbekistan', 'Vanuatu',
       'Venezuela (Bolivarian Republic of)', 'Viet Nam', 'Yemen',
       'Zambia', 'Zimbabwe']

# Streamlit UI
st.title("Life Expectancy Prediction")

# SideBar Descriptions for each feature 
st.sidebar.header("Input Feature Descriptions:")
st.sidebar.text("""
       GDP: This is the country's Gross Domestic Product per capita (in USD) e.g 380.52.\n
       Schooling: Schooling represents the number of years of education received by a person. (Input a numeric value ranging from 0 to 20, e.g 13.5).\n
       Income composition of resources: This is the Human Development Index in terms of income composition of resources (value ranging from 0 to 1, e.g 0.6).\n
       BMI: BMI is the Body Mass Index, representing a person's weight relative to their height. Input a numeric value (e.g., 15 to 50).\n
""")

st.write("Enter the features to predict life expectancy:")

# User input for country
country = st.selectbox("Select Country", countries)

# User input for other features
feature1 = st.number_input("Enter GDP", min_value=0.0, placeholder="Enter country's GDP")
feature2 = st.number_input("Enter Schooling Years", min_value=0.0, placeholder='Enter a value between 1-10')
feature3 = st.number_input("Income composition of resources", min_value=0.0, placeholder='Enter a value between 0-1')
feature4 = st.number_input("BMI", min_value=0.0, placeholder="Enter BMI")


# Prediction button
if st.button("Predict"):
    # Prepare the input data
    input_data = np.array([[feature1, feature2, feature3, feature4]]) 
    prediction = model.predict(input_data)
    
    # Display the predicted life expectancy
    st.write(f"Predicted Life Expectancy for {country}: {prediction[0]} years")


# Signature
st.markdown("---")  # Separator line
st.markdown(
    "<div style='text-align: right; font-size: 16px; font-style: italic;'>"
    "Signed, <br> Charles Ebong"
    "</div>",
    unsafe_allow_html=True
)
