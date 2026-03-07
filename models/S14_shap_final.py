import streamlit as st
import pickle
import pandas as pd
import numpy as np
import shap
import plotly.express as px

# -----------------------------
# LOAD MODEL AND TRAIN DATA
# -----------------------------
with open("xgb_spw3_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

X_train = pd.read_csv("../data/processed/X_train.csv")
X_test = pd.read_csv("../data/processed/X_test.csv")
y_train = pd.read_csv("../data/processed/y_train.csv")
y_test = pd.read_csv("../data/processed/y_test.csv")

# -----------------------------
# STREAMLIT PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="E-commerce Purchase Intent Prediction",
                   page_icon="🛒",
                   layout="wide")
st.title("Online Purchasing Prediction")
st.sidebar.header("Parameters")

# -----------------------------
# USER INPUT
# -----------------------------
def user_input_features():
    months = ['Aug','Dec','Feb','Jul','June','Mar','May','Nov','Oct','Sep']
    selected_month = st.sidebar.selectbox('Month', months, index=1)
    month_data = {f'month_{month}': 0 for month in months}
    month_data[f'month_{selected_month}'] = 1

    visitor_types = ['New_Visitor', 'Other', 'Returning_Visitor']
    selected_visitor = st.sidebar.selectbox('Visitor Type', visitor_types, index=0)
    visitor_data = {f'visitor_type_{v}': 1 if selected_visitor==v else 0 for v in visitor_types}

    os_types = [f'os_{i}' for i in range(1,9)]
    selected_os = st.sidebar.selectbox('Operating System', os_types, index=1)
    os_data = {f'os_{i}': 1 if selected_os==f'os_{i}' else 0 for i in range(1,9)}

    browsers = [f'browser_{i}' for i in range(1,14)]
    selected_browser = st.sidebar.selectbox('Browser', browsers, index=9)
    browser_data = {f'browser_{i}':1 if selected_browser==f'browser_{i}' else 0 for i in range(1,14)}

    regions = [f'region_{i}' for i in range(1,10)]
    selected_region = st.sidebar.selectbox('Region', regions, index=0)
    region_data = {f'region_{i}':1 if selected_region==f'region_{i}' else 0 for i in range(1,10)}

    traffic_types = [f'traffic_type_{i}' for i in range(1,21)]
    selected_traffic = st.sidebar.selectbox('Traffic Source', traffic_types, index=1)
    traffic_data = {f'traffic_type_{i}':1 if selected_traffic==f'traffic_type_{i}' else 0 for i in range(1,21)}

    bounce_rate_pct = st.sidebar.slider('Bounce Rate (%)',0,20,3)
    exit_rate_pct   = st.sidebar.slider('Exit Rate (%)',0,20,4)

    input_data = {
        'admin': st.sidebar.slider('Admin pages',0,27,7),
        'admin_duration': st.sidebar.slider('Admin duration',0,3400,139),
        'info': st.sidebar.slider('Info pages',0,24,0),
        'info_duration': st.sidebar.slider('Info duration',0,2600,0),
        'prod_related': st.sidebar.slider('Product pages viewed',0,705,30),
        'prod_related_duration': st.sidebar.slider('Product browsing time',0,70000,986),
        'bounce_rate': bounce_rate_pct/100,
        'exit_rate': exit_rate_pct/100,
        'page_value': st.sidebar.slider('Page value',0,362,36),
        'special_day': st.sidebar.selectbox('Special day',[0,1]),
        'weekend': st.sidebar.selectbox('Weekend',[0,1]),
        **month_data,
        **visitor_data,
        **os_data,
        **browser_data,
        **region_data,
        **traffic_data
    }
    return input_data

user_input = user_input_features()
input_df = pd.DataFrame([user_input]).astype(float)
feature_names = input_df.columns.tolist()

# -----------------------------
# PREDICTION
# -----------------------------
def predict(input_data):
    df = pd.DataFrame([input_data])
    pred = model.predict(df)
    prob = model.predict_proba(df)
    return pred, prob

prediction, probability = predict(user_input)
if probability[0][1] >= 0.58:
    st.success(f"✅ Purchase Made! — Confidence: {probability[0][1]*100:.1f}%")
else:
    st.warning(f"❌ No Purchase Made. — Confidence: {probability[0][0]*100:.1f}%")

st.subheader('Purchase Confidence')
st.progress(int(probability[0][1]*100))
st.caption(f"Model confidence in purchase: {probability[0][1]*100:.1f}%")

# -----------------------------
# SESSION PROFILING
# -----------------------------
st.divider()
st.header("🧩 Session Profiling")
weekend_value = "Yes" if user_input["weekend"] == 1 else "No"

def badge(value, color):
    return f'<span style="background:{color};padding:6px 12px;border-radius:12px;color:white;font-weight:600;">{value}</span>'

def red_badge(value):
    return f'<span style="background:#ef4444;padding:6px 12px;border-radius:12px;color:white;font-weight:600;">{value}</span>'

month = [k for k in feature_names if k.startswith('month_') and user_input[k]==1][0].replace("month_","")
visitor = [k for k in feature_names if k.startswith('visitor_type_') and user_input[k]==1][0].replace("visitor_type_","")
os = [k for k in feature_names if k.startswith('os_') and user_input[k]==1][0].replace("os_","")
browser = [k for k in feature_names if k.startswith('browser_') and user_input[k]==1][0].replace("browser_","")
region = [k for k in feature_names if k.startswith('region_') and user_input[k]==1][0].replace("region_","")
traffic = [k for k in feature_names if k.startswith('traffic_type_') and user_input[k]==1][0].replace("traffic_type_","")

st.markdown(f"""
<table style="width:100%; border-collapse:collapse;">
<tr>
<th>Visitor</th><th>Traffic</th><th>Region</th><th>Browser</th><th>OS</th><th>Weekend</th><th>Month</th>
</tr>
<tr>
<td>{badge(visitor,'#6366f1')}</td>
<td>{badge(traffic,'#0ea5e9')}</td>
<td>{badge(region,'#14b8a6')}</td>
<td>{badge(browser,'#8b5cf6')}</td>
<td>{badge(os,'#a855f7')}</td>
<td>{red_badge(weekend_value)}</td>
<td>{badge(month,'#f59e0b')}</td>
</tr>
</table>
""", unsafe_allow_html=True)

# -----------------------------
# SHAP EXPLAINER
# -----------------------------
st.divider()
st.header('📊 SHAP Feature Impact on Conversion')

background = X_train.sample(n=min(100, len(X_train)), random_state=42).astype(float)
explainer = shap.Explainer(lambda X: model.predict_proba(pd.DataFrame(X, columns=feature_names))[:,1], background.values)
shap_values = explainer(input_df.values)[0].values

# -----------------------------
# DEFINE FAMILIES
# -----------------------------
families = {}
numerical_features = ['admin','admin_duration','info','info_duration','prod_related','prod_related_duration',
                      'bounce_rate','exit_rate','page_value','special_day','weekend']
for f in numerical_features:
    if f in feature_names:
        families[f] = [f]

families['Month'] = [c for c in feature_names if c.startswith('month_')]
families['Visitor'] = [c for c in feature_names if c.startswith('visitor_type_')]
families['OS'] = [c for c in feature_names if c.startswith('os_')]
families['Browser'] = [c for c in feature_names if c.startswith('browser_')]
families['Region'] = [c for c in feature_names if c.startswith('region_')]
families['Traffic'] = [c for c in feature_names if c.startswith('traffic_type_')]

# -----------------------------
# AGGREGATE SHAP VALUES PER FAMILY
# -----------------------------
family_shap = {}
total_abs = np.sum(np.abs(shap_values))
for fam, feats in families.items():
    indices = [feature_names.index(f) for f in feats if f in feature_names]
    family_shap[fam] = 100*np.sum(shap_values[indices])/total_abs

family_df = pd.DataFrame({
    'Feature': list(family_shap.keys()),
    'SHAP Impact Rate (%)': list(family_shap.values())
})
family_df['Color'] = family_df['SHAP Impact Rate (%)'].apply(lambda x: 'positive' if x>=0 else 'negative')

# --------------------------------------------------
# SHAP FEATURE IMPACT PLOT (POS/NEG COLORS, PERCENTAGES ABOVE BARS)
# --------------------------------------------------
plot_df = family_df.copy()

fig = px.bar(
    plot_df,
    x="Feature",
    y="SHAP Impact Rate (%)",
    color="Color",
    color_discrete_map={"positive": "green", "negative": "red"},
    text="SHAP Impact Rate (%)",
    category_orders={"Feature": plot_df["Feature"].tolist()}
)

fig.update_traces(
    texttemplate="<b style='color:black;'>%{text:.1f}%</b>",
    textposition="outside",
    cliponaxis=False
)

# Extend y-axis to show negative bar labels
y_min = plot_df["SHAP Impact Rate (%)"].min() * 1.3 if plot_df["SHAP Impact Rate (%)"].min() < 0 else 0
y_max = plot_df["SHAP Impact Rate (%)"].max() * 1.3

fig.update_layout(
    xaxis_title="<b style='color:black;'>Feature</b>",
    yaxis_title="<b style='color:black;'>Impact Rate (%)</b>",
    xaxis=dict(tickfont=dict(family="Arial", size=12, color="black")),
    yaxis=dict(range=[y_min, y_max], tickfont=dict(family="Arial", size=12, color="black")),
    showlegend=True,
    legend_title_text="<b style='color:black;'>Impact Direction</b>"
)

st.plotly_chart(fig, use_container_width=True)