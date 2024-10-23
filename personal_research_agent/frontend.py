# frontend.py
import streamlit as st
import requests

# Streamlit application interface
st.title("Personal Research Assistant")
user_query = st.text_input("Enter your question:", "")

if st.button("Run Query"):
    if user_query:
        response = requests.post("http://localhost:8000/query/", json={"question": user_query})
        if response.status_code == 200:
            data = response.json()
            st.write(data['generation'])
        else:
            st.write("Error: Unable to fetch data.")