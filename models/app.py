import streamlit as st
import pickle
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

st.set_page_config(page_title="Online Purchase Prediction", layout="wide")

# -----------------------------
# Load model (cached for speed)
# -----------------------------
@st.cache_resource
def load_model():
    with open("xgb_spw3_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# -----------------------------
# Prediction function
# -----------------------------
def predict(input_data):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    probability = model.predict_proba(df)
    return prediction, probability


# -----------------------------
# Page Title
# -----------------------------
st.title("🛒 Online Purchasing Prediction")
st.markdown("Predict whether a website visitor will make a purchase.")

st.sidebar.header("User Session Parameters")

# -----------------------------
# User Inputs
# -----------------------------
def user_input_features():

    months = ['Aug','Dec','Feb','Jul','June','Mar','May','Nov','Oct','Sep']
    selected_month = st.sidebar.selectbox("Month", months)

    visitor_types = ['New_Visitor','Other','Returning_Visitor']
    selected_visitor = st.sidebar.selectbox("Visitor Type", visitor_types)

    os_types = [f'os_{i}' for i in range(1,9)]
    selected_os = st.sidebar.selectbox("Operating System", os_types)

    browsers = [f'browser_{i}' for i in range(1,14)]
    selected_browser = st.sidebar.selectbox("Browser", browsers)

    regions = [f'region_{i}' for i in range(1,10)]
    selected_region = st.sidebar.selectbox("Region", regions)

    traffic_types = [f'traffic_type_{i}' for i in range(1,21)]
    selected_traffic = st.sidebar.selectbox("Traffic Type", traffic_types)

    # numeric inputs
    admin = st.sidebar.slider("Administrative Pages",0,27,7)
    admin_duration = st.sidebar.slider("Administrative Duration",0,3400,139)

    info = st.sidebar.slider("Informational Pages",0,24,0)
    info_duration = st.sidebar.slider("Informational Duration",0,2600,0)

    prod_related = st.sidebar.slider("Product Pages",0,705,30)
    prod_related_duration = st.sidebar.slider("Product Page Duration",0,70000,986)

    bounce_rate = st.sidebar.slider("Bounce Rate %",0,20,0)/100
    exit_rate = st.sidebar.slider("Exit Rate %",0,20,1)/100

    page_value = st.sidebar.slider("Page Value",0,362,36)

    special_day = st.sidebar.selectbox("Special Day",[0,1])
    weekend = st.sidebar.selectbox("Weekend",[0,1])

    # One-hot encoding dictionaries
    month_data = {f'month_{m}':0 for m in months}
    month_data[f'month_{selected_month}']=1

    visitor_data = {f'visitor_type_{v}':0 for v in visitor_types}
    visitor_data[f'visitor_type_{selected_visitor}']=1

    os_data = {f'os_{i}':0 for i in range(1,9)}
    os_data[selected_os]=1

    browser_data = {f'browser_{i}':0 for i in range(1,14)}
    browser_data[selected_browser]=1

    region_data = {f'region_{i}':0 for i in range(1,10)}
    region_data[selected_region]=1

    traffic_data = {f'traffic_type_{i}':0 for i in range(1,21)}
    traffic_data[selected_traffic]=1

    input_data = {
        'admin':admin,
        'admin_duration':admin_duration,
        'info':info,
        'info_duration':info_duration,
        'prod_related':prod_related,
        'prod_related_duration':prod_related_duration,
        'bounce_rate':bounce_rate,
        'exit_rate':exit_rate,
        'page_value':page_value,
        'special_day':special_day,
        'weekend':weekend,
        **month_data,
        **visitor_data,
        **os_data,
        **browser_data,
        **region_data,
        **traffic_data
    }

    return input_data


user_input = user_input_features()

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict Purchase"):

    prediction, probability = predict(user_input)

    purchase_prob = probability[0][1]
    no_purchase_prob = probability[0][0]

    st.subheader("Prediction Result")

    if purchase_prob >= 0.58:
        st.success(f"✅ Purchase Likely — Confidence: {purchase_prob*100:.2f}%")
    else:
        st.warning(f"❌ No Purchase Likely — Confidence: {no_purchase_prob*100:.2f}%")

    # -----------------------------
    # Probability Chart
    # -----------------------------
    st.subheader("Prediction Probability")

    chart_data = pd.DataFrame({
        "Outcome":["No Purchase","Purchase"],
        "Probability":[no_purchase_prob,purchase_prob]
    })

    st.bar_chart(chart_data.set_index("Outcome"))

    # -----------------------------
    # Progress indicator
    # -----------------------------
    st.subheader("Purchase Confidence")
    st.progress(int(purchase_prob*100))

    st.write(f"Model confidence in purchase: **{purchase_prob*100:.2f}%**")