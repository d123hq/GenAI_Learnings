# frontend.py
import streamlit as st
import requests

# Streamlit application interface
st.title("Personal Research Assistant")
user_query = st.text_input("Enter your question:", "")

if st.button("Run Query"):
    if user_query:
        # Use the service name for backend URL
        backend_url = "http://backend:8000/query/"
        
        # Show loading spinner while waiting for the response
        with st.spinner("Processing..."):
            try:
                response = requests.post(backend_url, json={"question": user_query})
                response.raise_for_status()  # Raise an error for bad responses
                
                # If the request is successful, display the result
                data = response.json()
                st.write(data['generation'])
            except requests.exceptions.HTTPError as http_err:
                st.write(f"HTTP error occurred: {http_err}")
            except requests.exceptions.RequestException as req_err:
                st.write(f"Error occurred: {req_err}")
            except Exception as e:
                st.write(f"An unexpected error occurred: {e}")
    else:
        st.write("Please enter a question before running the query.")


# import streamlit as st
# import requests

# # Streamlit application interface
# st.title("Personal Research Assistant")
# user_query = st.text_input("Enter your question:", "")

# if st.button("Run Query"):
#     if user_query:
#         response = requests.post("http://127.0.01:8000/query/", json={"question": user_query})
#         if response.status_code == 200:
#             data = response.json()
#             st.write(data['generation'])
#         else:
#             st.write("Error: Unable to fetch data.")