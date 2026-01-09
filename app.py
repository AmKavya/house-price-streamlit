import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ------------------ LOAD MODEL ------------------
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="House Price Predictor",
    layout="centered"
)

# ------------------ TITLE ------------------
st.title("House Price Prediction")
st.caption("Estimate house prices using a trained Random Forest model")

st.markdown("---")

# ------------------ INPUT SECTION ------------------
st.subheader("House Characteristics")

col1, col2 = st.columns(2)

with col1:
    RM = st.slider(
        "Average number of rooms (RM)",
        min_value=1.0,
        max_value=10.0,
        value=6.0
    )

    LSTAT = st.slider(
        "Lower status population (%) (LSTAT)",
        min_value=0.0,
        max_value=40.0,
        value=12.0
    )

with col2:
    CRIM = st.slider(
        "Crime rate per capita (CRIM)",
        min_value=0.0,
        max_value=90.0,
        value=0.2
    )

    CHAS = st.selectbox(
        "Charles River proximity (CHAS)",
        options=[0, 1],
        format_func=lambda x: "Yes (bounds river)" if x == 1 else "No"
    )

# ------------------ ADVANCED DEFAULTS ------------------
with st.expander("Advanced Assumptions (Fixed Defaults)"):
    st.write("These features are kept constant based on average Boston housing data:")
    st.markdown("""
    - **ZN**: 0.0  
    - **INDUS**: 0.0  
    - **NOX**: 0.5  
    - **AGE**: 60  
    - **DIS**: 5  
    - **RAD**: 4  
    - **TAX**: 330  
    - **PTRATIO**: 18  
    - **B**: 390  
    """)

# ------------------ INPUT DATA ------------------
input_data = pd.DataFrame([{
    "CRIM": CRIM,
    "ZN": 0.0,
    "INDUS": 0.0,
    "CHAS": CHAS,
    "NOX": 0.5,
    "RM": RM,
    "AGE": 60,
    "DIS": 5.0,
    "RAD": 4,
    "TAX": 330,
    "PTRATIO": 18.0,
    "B": 390.0,
    "LSTAT": LSTAT
}])

input_prepared = pipeline.transform(input_data)

# ------------------ PREDICTION ------------------
st.markdown("---")

if st.button("Predict House Price"):
    prediction = model.predict(input_prepared)[0]

    st.metric(
        label="Estimated House Price",
        value=f"${prediction*1000:,.0f}"
    )

    with st.expander("How to interpret this result"):
        st.write("""
        - This prediction is based on historical Boston housing data.
        - Some features are fixed at average values.
        - This is an **estimate**, not a market guarantee.
        """)

# ------------------ SIDEBAR ------------------
st.sidebar.header("Model Information")
st.sidebar.write("**Algorithm:** Random Forest Regressor")
st.sidebar.write("**User Inputs:**")
st.sidebar.write("""
- RM (Rooms)  
- CRIM (Crime Rate)  
- LSTAT (%)  
- CHAS  
""")

st.sidebar.markdown("---")
st.sidebar.caption("Built with using Streamlit & Scikit-learn")



























# import streamlit as st
# import numpy as np
# import pandas as pd
# import seaborn as sns 
# import matplotlib.pyplot as plt 
# import joblib

# model = joblib.load("model.pkl")
# pipeline = joblib.load("pipeline.pkl")


# st.title("House Price Prediction")
# st.write("Predict house prices using Random Forest")

# st.subheader("House Characteristics")

# RM = st.slider(
#     "Average number of rooms per dwelling (RM)",
#     min_value=1.0,
#     max_value=10.0,
#     value=6.0
# )

# CRIM = st.slider(
#     "Per capita crime rate by town (CRIM)",
#     min_value=0.0,
#     max_value=90.0,
#     value=0.2
# )

# LSTAT = st.slider(
#     "Percentage of lower status population (LSTAT)",
#     min_value=0.0,
#     max_value=40.0,
#     value=12.0
# )


# CHAS = st.selectbox(
#     "Charles River proximity (CHAS)",
#     options=[0, 1],
#     format_func=lambda x: "Yes (bounds river)" if x == 1 else "No"
# )

# # TAXRM = st.slider(
# #     "Property tax per room (TAX / RM)",
# #     min_value=10.0,
# #     max_value=150.0,
# #     value=50.0
# # )



# input_data = pd.DataFrame([{
#     "CRIM": CRIM,
#     "ZN": 0.0,         # default
#     "INDUS": 0.0,      # default
#     "CHAS": CHAS,
#     "NOX": 0.5,        # default
#     "RM": RM,
#     "AGE": 60,         # default
#     "DIS": 5.0,        # default
#     "RAD": 4,          # default
#     # "TAXRM": TAXRM,
#     "TAX":330,         # default
#     "PTRATIO": 18.0,   # default
#     "B": 390.0,        # default
#     "LSTAT": LSTAT
# }])


# input_prepared = pipeline.transform(input_data)


# if st.button("Predict Price"):
#     prediction = model.predict(input_prepared)
#     st.success(f"Predicted House Price: ${prediction[0]*1000:.2f}")


