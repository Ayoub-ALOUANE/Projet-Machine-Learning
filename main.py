import streamlit as st
import importlib

# Read URL parameter ?app=car or ?app=house
query_params = st.query_params
app_param = query_params.get("app", "house")
# Handle both single value and list formats
if isinstance(app_param, list):
    app_to_load = app_param[0] if app_param else "house"
else:
    app_to_load = app_param

# Set page config based on app
page_title = "Car Predict Pro" if app_to_load == "car" else "House Predict Pro"
st.set_page_config(page_title=page_title, layout="wide")

# Load and run the selected app
if app_to_load == "car":
    # Show navigation to house app
    if st.button("🏠 Switch to House Predict Pro", use_container_width=True):
        st.query_params["app"] = "house"
        st.rerun()

    module = importlib.import_module("Cars.app")
else:
    # Show navigation to car app
    if st.button("🚗 Switch to Car Predict Pro", use_container_width=True):
        st.query_params["app"] = "car"
        st.rerun()

    module = importlib.import_module("Housing.app")

module.run()