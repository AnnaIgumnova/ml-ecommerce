import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import shap
import xgboost as xgb
import numpy as np

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="E-commerce Purchase Intent Prediction",
    page_icon="🛒",
    layout="wide"
)
st.title("Online Purchasing Prediction")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
with open("xgb_spw3_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# --------------------------------------------------
# NUMERICAL FEATURES ORDER
# --------------------------------------------------
NUMERICAL_COLUMNS_ORDER = [
    'admin',
    'admin_duration',
    'info',
    'info_duration',
    'prod_related',
    'prod_related_duration',
    'bounce_rate',
    'exit_rate',
    'page_value',
    'special_day'
]

# --------------------------------------------------
# FUNCTIONS
# --------------------------------------------------
def predict(data):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    probability = model.predict_proba(df)
    return prediction, probability

# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------
st.sidebar.header("Parameters")
def user_input():
    months = ['Aug','Dec','Feb','Jul','June','Mar','May','Nov','Oct','Sep']
    month = st.sidebar.selectbox("Month", months)
    month_data = {f"month_{m}":0 for m in months}
    month_data[f"month_{month}"] = 1

    visitors = ['New_Visitor','Other','Returning_Visitor']
    visitor = st.sidebar.selectbox("Visitor type", visitors)
    visitor_data = {f'visitor_type_{v}':1 if v==visitor else 0 for v in visitors}

    os_types = [f"os_{i}" for i in range(1,9)]
    os = st.sidebar.selectbox("Operating system", os_types)
    os_data = {f"os_{i}":1 if os==f"os_{i}" else 0 for i in range(1,9)}

    browsers = [f"browser_{i}" for i in range(1,14)]
    browser = st.sidebar.selectbox("Browser", browsers)
    browser_data = {f"browser_{i}":1 if browser==f"browser_{i}" else 0 for i in range(1,14)}

    regions = [f"region_{i}" for i in range(1,10)]
    region = st.sidebar.selectbox("Region", regions)
    region_data = {f"region_{i}":1 if region==f"region_{i}" else 0 for i in range(1,10)}

    traffic_types = [f"traffic_type_{i}" for i in range(1,21)]
    traffic = st.sidebar.selectbox("Traffic source", traffic_types)
    traffic_data = {f"traffic_type_{i}":1 if traffic==f"traffic_type_{i}" else 0 for i in range(1,21)}

    bounce = st.sidebar.slider("Bounce rate (%)",0,20,3)
    exit_rate = st.sidebar.slider("Exit rate (%)",0,20,4)

    data = {
        'admin': st.sidebar.slider("Admin pages",0,27,7),
        'admin_duration': st.sidebar.slider("Admin duration",0,3400,139),
        'info': st.sidebar.slider("Info pages",0,24,0),
        'info_duration': st.sidebar.slider("Info duration",0,2600,0),
        'prod_related': st.sidebar.slider("Product pages viewed",0,705,30),
        'prod_related_duration': st.sidebar.slider("Product browsing time",0,70000,986),
        'bounce_rate': float(bounce/100),
        'exit_rate': float(exit_rate/100),
        'page_value': st.sidebar.slider("Page value",0,362,36),
        'special_day': st.sidebar.selectbox("Special day",[0,1]),
        'weekend': st.sidebar.selectbox("Weekend",[0,1]),
        **month_data,
        **visitor_data,
        **os_data,
        **browser_data,
        **region_data,
        **traffic_data
    }
    return data, month, visitor, browser, os, region, traffic

data, month, visitor, browser, os, region, traffic = user_input()

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
prediction, probability = predict(data)
if probability[0][1] >= 0.58:
    confidence = probability[0][1] * 100
    st.success(f"✅ Purchase Made! — Confidence: {confidence:.1f}%")
else:
    confidence = probability[0][0] * 100
    st.warning(f"❌ No Purchase Made. — Confidence: {confidence:.1f}%")

st.subheader("Purchase Confidence")
st.progress(int(probability[0][1] * 100))
st.caption(f"Model confidence in purchase: {probability[0][1]*100:.1f}%")

# --------------------------------------------------
# SESSION PROFILING
# --------------------------------------------------
st.divider()
st.header("🧩 Session Profiling")
weekend_value = "Yes" if data["weekend"] == 1 else "No"

def badge(value, color):
    return f'<span style="background:{color};padding:6px 12px;border-radius:12px;color:white;font-weight:600;">{value}</span>'
def red_badge(value):
    return f'<span style="background:#ef4444;padding:6px 12px;border-radius:12px;color:white;font-weight:600;">{value}</span>'

st.markdown(f"""
<table style="width:100%; border-collapse:collapse;">
<thead>
<tr>
<th>Visitor</th>
<th>Traffic</th>
<th>Region</th>
<th>Browser</th>
<th>OS</th>
<th>Weekend</th>
<th>Month</th>
</tr>
</thead>
<tbody>
<tr>
<td>{badge(visitor,"#6366f1")}</td>
<td>{badge(traffic,"#0ea5e9")}</td>
<td>{badge(region,"#14b8a6")}</td>
<td>{badge(browser,"#8b5cf6")}</td>
<td>{badge(os,"#a855f7")}</td>
<td>{red_badge(weekend_value)}</td>
<td>{badge(month,"#f59e0b")}</td>
</tr>
</tbody>
</table>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SHAP HORIZONTAL BAR PLOT (NUMERICAL FEATURES)
# --------------------------------------------------
st.divider()
st.header("📈 SHAP Feature Impact (Numerical Features Only)")

# Use all features for SHAP
full_data = pd.DataFrame([data]).astype(float)
full_feature_names = full_data.columns.tolist()

# Callable wrapper for SHAP
def model_predict(X):
    df = pd.DataFrame(X, columns=full_feature_names)
    return model.predict_proba(df)[:, 1]

explainer = shap.Explainer(model_predict, full_data.values)
shap_values_full = explainer(full_data.values)[0].values

# Keep only numerical features for plotting
shap_values_num = shap_values_full[:len(NUMERICAL_COLUMNS_ORDER)]
numerical_data_for_plot = full_data[NUMERICAL_COLUMNS_ORDER]

shap_df = pd.DataFrame({
    'Feature': NUMERICAL_COLUMNS_ORDER,
    'SHAP Value': shap_values_num
})
shap_df['Color'] = shap_df['SHAP Value'].apply(lambda x: 'green' if x >=0 else 'red')
shap_df['AbsValue'] = shap_df['SHAP Value'].abs()
shap_df = shap_df.sort_values('AbsValue', ascending=True)

fig = px.bar(
    shap_df,
    x='SHAP Value',
    y='Feature',
    orientation='h',
    color='Color',
    color_discrete_map={'green':'green','red':'red'},
    text='SHAP Value'
)
fig.update_traces(texttemplate="<b style='color:black;'>%{text:.3f}</b>", textposition='outside')
fig.update_layout(
    xaxis_title="<b>SHAP Value</b>",
    yaxis_title="<b>Feature</b>",
    yaxis=dict(autorange="reversed"),
    showlegend=False
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# SHAP SUMMARY SCATTER PLOT (NUMERICAL FEATURES)
# --------------------------------------------------
st.divider()
st.header("📊 SHAP Summary Plot (Numerical Features)")

# Prepare DataFrame for scatter
shap_summary_df = pd.DataFrame(shap_values_full, columns=full_feature_names)
shap_summary_df = shap_summary_df[NUMERICAL_COLUMNS_ORDER]  # only numerical
shap_summary_df = shap_summary_df.reset_index().melt(id_vars='index', var_name='Feature', value_name='SHAP Value')

# Add feature values for coloring
feature_values_df = numerical_data_for_plot.reset_index().melt(id_vars='index', var_name='Feature', value_name='Feature Value')
shap_summary_df['Feature Value'] = feature_values_df['Feature Value']

# Plotly scatter
fig_summary = px.strip(
    shap_summary_df,
    x='SHAP Value',
    y='Feature',
    color='Feature Value',
    color_continuous_scale=px.colors.sequential.Viridis,
    stripmode='overlay',
    hover_data={'Feature Value': True, 'SHAP Value': ':.3f'}
)
fig_summary.update_layout(
    xaxis_title="<b>SHAP Value</b>",
    yaxis_title="<b>Feature</b>",
    yaxis=dict(autorange="reversed"),
    coloraxis_colorbar=dict(title="Feature Value"),
    height=600
)
st.plotly_chart(fig_summary, use_container_width=True)