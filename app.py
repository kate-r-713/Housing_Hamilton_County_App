import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import openpyxl
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ----------------------------
# Title & Description
# ----------------------------
st.title("Hamilton County Residential Property Value Predictor")
st.write(
    """
    This app predicts residential property **APPRAISED_VALUE**
    using a Linear Regression model trained on historical assessor data.
    """
)

# ----------------------------
# Load and Preprocess Data
# ----------------------------
@st.cache_data
def load_and_train_model():
    url = "Housing_Hamilton_County.xlsx"
    data = pd.read_excel(url, sheet_name='AssessorExport')

    # Rename columns
    data.rename(columns={'APPRAISED_VALUE': 'AP'}, inplace=True)
    data.rename(columns={'LAND_USE_CODE_DESC': 'LAND'}, inplace=True)

    # Convert AP to numeric
    data['AP'] = pd.to_numeric(data['AP'], errors='coerce')
    data_clean = data.dropna(subset=['AP'])

    # Keep residential only
    residential_data = data_clean[data_clean['LAND'] == 'RESIDENTIAL']

    # Standardize column names
    residential_data.columns = (
        residential_data.columns
        .str.strip()
        .str.upper()
        .str.replace(" ", "_")
    )

    residential_only = residential_data[
        residential_data['LAND'].str.upper() == 'RESIDENTIAL'
    ]

    residential_only['AP'] = pd.to_numeric(
        residential_only['AP'], errors='coerce'
    )

    features = [
        'CALC_ACRES',
        'NEIGHBORHOOD_CODE',
        'DISTRICT',
        'PROPERTY_TYPE_CODE',
        'CURRENT_USE_CODE',
        'ZONING'
    ]

    model_data = residential_only[features + ['AP']].dropna()

    X = model_data[features]
    y = model_data['AP']

    categorical_cols = [
        'NEIGHBORHOOD_CODE',
        'DISTRICT',
        'PROPERTY_TYPE_CODE',
        'CURRENT_USE_CODE',
        'ZONING'
    ]

    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X.columns, model_data


model, model_columns, model_data = load_and_train_model()

# ----------------------------
# User Inputs
# ----------------------------
st.header("Enter Property Information")

calc_acres = st.number_input("Lot Size (Acres)", min_value=0.0, value=0.25)

neighborhood = st.selectbox(
    "Neighborhood Code",
    sorted(model_data['NEIGHBORHOOD_CODE'].unique())
)

district = st.selectbox(
    "District",
    sorted(model_data['DISTRICT'].unique())
)

property_type = st.selectbox(
    "Property Type Code",
    sorted(model_data['PROPERTY_TYPE_CODE'].unique())
)

current_use = st.selectbox(
    "Current Use Code",
    sorted(model_data['CURRENT_USE_CODE'].unique())
)

zoning = st.selectbox(
    "Zoning",
    sorted(model_data['ZONING'].unique())
)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Appraised Value"):

    input_dict = {
        'CALC_ACRES': calc_acres,
        'NEIGHBORHOOD_CODE': neighborhood,
        'DISTRICT': district,
        'PROPERTY_TYPE_CODE': property_type,
        'CURRENT_USE_CODE': current_use,
        'ZONING': zoning
    }

    input_df = pd.DataFrame([input_dict])

    categorical_cols = [
        'NEIGHBORHOOD_CODE',
        'DISTRICT',
        'PROPERTY_TYPE_CODE',
        'CURRENT_USE_CODE',
        'ZONING'
    ]

    input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

    # Align with training columns
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_df)[0]

    st.subheader("Predicted Appraised Value:")
    st.success(f"${prediction:,.2f}")

# ----------------------------
# Disclaimer
# ----------------------------
st.markdown("---")
st.caption("Disclaimer: This model is for educational purposes only. "
           "Predictions are estimates and should not be used for official property valuation.")
