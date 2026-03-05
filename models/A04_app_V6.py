import streamlit as st
import pickle
import xgboost as xgb
import pandas as pd

# Load the model
with open("xgb_spw3_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Function to make predictions using the loaded model
def predict(input_data):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    return prediction, probability

# Streamlit interface
st.title('Online Purchasing Prediction')

# Input fields for the user to enter data
st.sidebar.header('Parameters')

# Define the input form
def user_input_features():
    months = ['Aug', 'Dec', 'Feb', 'Jul', 'June', 'Mar', 'May', 'Nov', 'Oct', 'Sep']
    selected_month = st.sidebar.selectbox('Select Month', months, index=1)

    month_data = {f'month_{month}': 0 for month in months}
    month_data[f'month_{selected_month}'] = 1

    visitor_types = ['New_Visitor', 'Other', 'Returning_Visitor']
    selected_visitor = st.sidebar.selectbox('Select Visitor Type', visitor_types, index=0)
    visitor_data = {f'visitor_type_{visitor}': 1 if selected_visitor == visitor else 0 for visitor in visitor_types}

    os_types = [f'os_{i}' for i in range(1, 9)]
    selected_os = st.sidebar.selectbox('Select OS', os_types, index=1)
    os_data = {f'os_{i}': 1 if selected_os == f'os_{i}' else 0 for i in range(1, 9)}

    browsers = [f'browser_{i}' for i in range(1, 14)]
    selected_browser = st.sidebar.selectbox('Select Browser', browsers, index=9)
    browser_data = {f'browser_{i}': 1 if selected_browser == f'browser_{i}' else 0 for i in range(1, 14)}

    regions = [f'region_{i}' for i in range(1, 10)]
    selected_region = st.sidebar.selectbox('Select Region', regions, index=0)
    region_data = {f'region_{i}': 1 if selected_region == f'region_{i}' else 0 for i in range(1, 10)}

    traffic_types = [f'traffic_type_{i}' for i in range(1, 21)]
    selected_traffic = st.sidebar.selectbox('Select Traffic Type', traffic_types, index=1)
    traffic_data = {f'traffic_type_{i}': 1 if selected_traffic == f'traffic_type_{i}' else 0 for i in range(1, 21)}

    bounce_rate_pct = st.sidebar.slider('Bounce Rate (%)', 0, 20, 0, step=1)
    exit_rate_pct   = st.sidebar.slider('Exit Rate (%)',   0, 20, 1, step=1)

    input_data = {
        'admin': st.sidebar.slider('Administrative pages visited', 0, 27, value=7),
        'admin_duration': st.sidebar.slider('Administrative pages Duration (seconds)', 0, 3400, value=139),
        'info': st.sidebar.slider('Informational pages visited', 0, 24, value=0),
        'info_duration': st.sidebar.slider('Informational pages Duration (seconds)', 0, 2600, value=0),
        'prod_related': st.sidebar.slider('Product Related pages visited', 0, 705, value=30),
        'prod_related_duration': st.sidebar.slider('Product Related pages Duration (seconds)', 0, 70000, value=986),
        'bounce_rate': bounce_rate_pct / 100,
        'exit_rate':   exit_rate_pct   / 100,
        'page_value': st.sidebar.slider('Page Value', 0, 362, value=36),
        'special_day': st.sidebar.selectbox('Special Day', [0, 1], index=0),
        'weekend': st.sidebar.selectbox('Weekend', [0, 1], index=0),
        **month_data,
        **visitor_data,
        **os_data,
        **browser_data,
        **region_data,
        **traffic_data,
    }
    return input_data

# Collect user input
user_input = user_input_features()

# Prediction
prediction, probability = predict(user_input)

# Display prediction

if probability[0][1] >= 0.58:
    confidence = probability[0][1] * 100
    st.success(f"✅ Purchase Made! — Confidence: {confidence:.1f}%")
else:
    confidence = probability[0][0] * 100
    st.warning(f"❌ No Purchase Made. — Confidence: {confidence:.1f}%")



# Visual probability bar
st.subheader('Purchase Confidence')
st.progress(int(probability[0][1] * 100))
st.caption(f"Model confidence in purchase: {probability[0][1]*100:.1f}%")
