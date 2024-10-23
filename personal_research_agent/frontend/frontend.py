# frontend.py
import streamlit as st
import requests

# Streamlit application interface
st.title("Personal Research Assistant")
user_query = st.text_input("Enter your question:", "")

# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if st.button("Run Query"):
    if user_query:
        # Use the service name for backend URL
        backend_url = "http://backend:8000/query/"
        
        # Show loading spinner while waiting for the response
        with st.spinner("Processing..."):
            try:
                response = requests.post(backend_url, json={"question": user_query})
                response.raise_for_status()  # Raise an error for bad responses
                
                # If the request is successful, update chat history
                data = response.json()
                
                # Append the latest question and answer to session state
                st.session_state.chat_history.append({"user": user_query, "assistant": data['generation']})
                
                # Display updated chat history
                for chat in st.session_state.chat_history:
                    st.write(f"**You:** {chat['user']}")
                    st.write(f"**Assistant:** {chat['assistant']}")
                    
            except requests.exceptions.HTTPError as http_err:
                st.write(f"HTTP error occurred: {http_err}")
            except requests.exceptions.RequestException as req_err:
                st.write(f"Error occurred: {req_err}")
            except Exception as e:
                st.write(f"An unexpected error occurred: {e}")
    else:
        st.write("Please enter a question before running the query.")
