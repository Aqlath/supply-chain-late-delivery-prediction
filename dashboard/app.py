
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st



# 1. PAGE CONFIGURATION

st.set_page_config(
    page_title="Supply Chain Risk Predictor",
    page_icon="📦",
    layout="wide"
)

PROJECT_DIR = Path(__file__).resolve().parent.parent

REPORTS_DIR = PROJECT_DIR / "reports"

PROCESSED_DATA_PATH = (
    PROJECT_DIR
    / "datasets"
    / "processed"
    / "DataCoSupplyChain_ML_ready.csv"
)

API_URL = "http://127.0.0.1:8000"


# 2. LOAD DATA


@st.cache_data
def load_reports():
    shipment_df = pd.read_csv(
        REPORTS_DIR / "shipment_risk_priority_report.csv"
    )

    regional_df = pd.read_csv(
        REPORTS_DIR / "regional_risk_analysis.csv"
    )

    type_df = pd.read_csv(
        REPORTS_DIR / "shipment_type_risk_analysis.csv"
    )

    return shipment_df, regional_df, type_df


@st.cache_data
def load_dataset():
    return pd.read_csv(PROCESSED_DATA_PATH)


st.title("📦 Smart Supply Chain Risk Predictor")

st.caption(
    "Predict late-delivery risk, understand shipment-specific "
    "risk indicators, and support operational decisions."
)

try:
    shipment_df, regional_df, type_df = load_reports()
    dataset = load_dataset()

except FileNotFoundError as error:
    st.error(f"Required file not found: {error}")
    st.stop()


# 3. NAVIGATION


page = st.radio(
    "Navigation",
    ["Dashboard", "Predict Shipment"],
    horizontal=True,
    label_visibility="collapsed"
)



# 4. DASHBOARD


if page == "Dashboard":

    
    # SIDEBAR FILTERS
    

    st.sidebar.header("Shipment Filters")

    risk_options = [
        "High Risk",
        "Medium Risk",
        "Low Risk"
    ]

    selected_risks = st.sidebar.multiselect(
        "Risk categories",
        options=risk_options,
        default=risk_options
    )

    all_regions = sorted(
        shipment_df["Order Region"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_regions = st.sidebar.multiselect(
        "Order regions",
        options=all_regions,
        default=[]
    )

    filtered_df = shipment_df[
        shipment_df["Risk_Category"].isin(selected_risks)
    ].copy()

    if selected_regions:
        filtered_df = filtered_df[
            filtered_df["Order Region"].isin(selected_regions)
        ]

    
    # DASHBOARD METRICS
    

    st.header("📊 Shipment Risk Overview")

    total_shipments = len(filtered_df)

    high_risk_count = (
        filtered_df["Risk_Category"] == "High Risk"
    ).sum()

    average_risk = (
        filtered_df["Late_Delivery_Probability"].mean()
        if total_shipments > 0
        else 0
    )

    high_risk_rate = (
        high_risk_count / total_shipments * 100
        if total_shipments > 0
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Shipments in View",
        f"{total_shipments:,}"
    )

    c2.metric(
        "High-Risk Shipments",
        f"{high_risk_count:,}"
    )

    c3.metric(
        "High-Risk Rate",
        f"{high_risk_rate:.2f}%"
    )

    c4.metric(
        "Average Predicted Risk",
        f"{average_risk:.2%}"
    )

   
    # RISK CATEGORY CHART
    

    st.subheader("Shipment Count by Risk Category")

    if not filtered_df.empty:

        risk_counts = (
            filtered_df["Risk_Category"]
            .value_counts()
            .reindex(risk_options, fill_value=0)
            .rename_axis("Risk Category")
            .reset_index(name="Shipments")
        )

        fig_risk = px.bar(
            risk_counts,
            x="Risk Category",
            y="Shipments",
            color="Risk Category",
            text="Shipments",
            title="Shipment Distribution by Risk Category"
        )

        st.plotly_chart(
            fig_risk,
            use_container_width=True
        )

    else:
        st.info("No shipments match the selected filters.")

   
    # REGIONAL ANALYSIS
    

    st.subheader("Regional Risk Analysis")

    if not regional_df.empty:

        regional_chart = regional_df.sort_values(
            "High_Risk_Rate_Percent",
            ascending=False
        ).head(10)

        fig_region = px.bar(
            regional_chart.sort_values(
                "High_Risk_Rate_Percent"
            ),
            x="High_Risk_Rate_Percent",
            y="Order Region",
            orientation="h",
            title="Top 10 Regions by High-Risk Rate",
            labels={
                "High_Risk_Rate_Percent": "High-Risk Rate (%)",
                "Order Region": "Region"
            }
        )

        st.plotly_chart(
            fig_region,
            use_container_width=True
        )

    else:
        st.info("Regional analysis data is unavailable.")

   
    # SHIPMENT TYPE ANALYSIS
   

    st.subheader("Risk by Shipment Type")

    if not type_df.empty:

        fig_type = px.bar(
            type_df,
            x="Type",
            y="High_Risk_Rate_Percent",
            color="Type",
            text="High_Risk_Rate_Percent",
            title="High-Risk Rate by Shipment Type",
            labels={
                "High_Risk_Rate_Percent": "High-Risk Rate (%)",
                "Type": "Shipment Type"
            }
        )

        st.plotly_chart(
            fig_type,
            use_container_width=True
        )

    else:
        st.info("Shipment-type analysis data is unavailable.")

    # SHIPMENT PRIORITY TABLE
   

    st.subheader("Shipment Priority List")

    display_columns = [
        "Priority_Rank",
        "Original_Row_Index",
        "Late_Delivery_Probability",
        "Risk_Category",
        "Days for shipment (scheduled)",
        "Order Item Quantity",
        "Sales",
        "Order Region",
        "Type"
    ]

    available_columns = [
        column for column in display_columns
        if column in filtered_df.columns
    ]

    if not filtered_df.empty:

        st.dataframe(
            filtered_df[available_columns].head(100),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("There are no shipments to display.")

    st.download_button(
        "Download Filtered Shipment Report",
        data=filtered_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="filtered_shipment_risk_report.csv",
        mime="text/csv"
    )


# 5. HELPER FUNCTIONS FOR PREDICTION


def get_options(column):
    """Return the actual available values from the processed dataset."""

    values = dataset[column].dropna().unique().tolist()

    if not values:
        return []

    try:
        return sorted(values)

    except TypeError:
        return sorted(values, key=str)


categorical_columns = [
    "Type",
    "Customer Country",
    "Customer Segment",
    "Market",
    "Customer State",
    "Order Region",
    "Order Country",
    "Category Id",
    "Product Card Id",
    "Department Id"
]

missing_columns = [
    column for column in categorical_columns
    if column not in dataset.columns
]



# 6. PREDICT SHIPMENT


if page == "Predict Shipment":

    st.divider()

    st.header("🔮 Predict Risk for a New Shipment")

    st.write(
        "Enter shipment details to obtain an individual prediction, "
        "plain-English explanations of model risk indicators, "
        "and practical investigation recommendations."
    )

    if missing_columns:

        st.error(
            "The processed dataset is missing these columns: "
            + ", ".join(missing_columns)
        )

        st.stop()

    # SHIPMENT INPUT FORM


    with st.form("shipment_prediction_form"):

        st.subheader("Numerical Features")

        c1, c2, c3 = st.columns(3)

        with c1:

            scheduled_days = st.number_input(
                "Scheduled shipment days",
                min_value=0,
                value=1
            )

            discount = st.number_input(
                "Order item discount",
                min_value=0.0,
                value=10.0
            )

            quantity = st.number_input(
                "Order item quantity",
                min_value=1,
                value=2
            )

            sales = st.number_input(
                "Sales",
                min_value=0.0,
                value=99.98
            )

        with c2:

            product_price = st.number_input(
                "Product price",
                min_value=0.0,
                value=49.99
            )

            latitude = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=18.0
            )

            longitude = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=75.0
            )

            order_year = st.number_input(
                "Order year",
                min_value=1900,
                max_value=2100,
                value=2017
            )

        with c3:

            order_month = st.number_input(
                "Order month",
                min_value=1,
                max_value=12,
                value=1
            )

            order_day = st.number_input(
                "Order day",
                min_value=1,
                max_value=31,
                value=5
            )

            order_day_of_week = st.number_input(
                "Order day of week (0 = Monday, 6 = Sunday)",
                min_value=0,
                max_value=6,
                value=4
            )

            order_hour = st.number_input(
                "Order hour (0–23)",
                min_value=0,
                max_value=23,
                value=10
            )

        st.subheader("Categorical Features")

        d1, d2, d3 = st.columns(3)

        with d1:

            order_type = st.selectbox(
                "Shipment Type",
                get_options("Type")
            )

            customer_country = st.selectbox(
                "Customer Country",
                get_options("Customer Country")
            )

            customer_segment = st.selectbox(
                "Customer Segment",
                get_options("Customer Segment")
            )

            market = st.selectbox(
                "Market",
                get_options("Market")
            )

        with d2:

            customer_state = st.selectbox(
                "Customer State",
                get_options("Customer State")
            )

            order_region = st.selectbox(
                "Order Region",
                get_options("Order Region")
            )

            order_country = st.selectbox(
                "Order Country",
                get_options("Order Country")
            )

        with d3:

            category_id = st.selectbox(
                "Category ID",
                get_options("Category Id")
            )

            product_card_id = st.selectbox(
                "Product Card ID",
                get_options("Product Card Id")
            )

            department_id = st.selectbox(
                "Department ID",
                get_options("Department Id")
            )

        submitted = st.form_submit_button(
            "Predict Delivery Risk",
            use_container_width=True
        )

 
    # SEND INPUT TO FASTAPI
   

    if submitted:

        payload = {
            "days_for_shipment_scheduled": int(scheduled_days),
            "order_item_discount": float(discount),
            "order_item_quantity": int(quantity),
            "sales": float(sales),
            "product_price": float(product_price),
            "latitude": float(latitude),
            "longitude": float(longitude),
            "order_year": int(order_year),
            "order_month": int(order_month),
            "order_day": int(order_day),
            "order_day_of_week": int(order_day_of_week),
            "order_hour": int(order_hour),
            "type": str(order_type),
            "customer_country": str(customer_country),
            "customer_segment": str(customer_segment),
            "market": str(market),
            "customer_state": str(customer_state),
            "order_region": str(order_region),
            "order_country": str(order_country),
            "category_id": int(category_id),
            "product_card_id": int(product_card_id),
            "department_id": int(department_id)
        }

        try:

            with st.spinner(
                "Predicting risk and generating shipment-specific explanations..."
            ):

                response = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=60
                )

            response.raise_for_status()

            result = response.json()

         
            # PREDICTION SUMMARY
           

            st.divider()

            st.header("📋 Shipment Prediction Result")

            probability = result["late_delivery_probability"]
            risk_category = result["risk_category"]

            r1, r2, r3 = st.columns(3)

            r1.metric(
                "Late-Delivery Probability",
                f"{probability:.2%}"
            )

            r2.metric(
                "Predicted Late Risk",
                "Yes" if result["predicted_late_risk"] else "No"
            )

            r3.metric(
                "Risk Category",
                risk_category
            )

            if risk_category == "High Risk":
                st.error(result["message"])

            elif risk_category == "Medium Risk":
                st.warning(result["message"])

            else:
                st.success(result["message"])

            st.caption(
                f"Classification threshold: "
                f"{result['classification_threshold']:.2f}"
            )

          
            # POSSIBLE RISK FACTORS
          

            st.header("🔍 Possible Risk Factors")

            risk_factors = result.get("risk_factors", [])

            if risk_factors:

                st.caption(
                    "These recorded input values were associated by "
                    "the model with higher late-delivery risk. The "
                    "associations are not confirmed causes of delay."
                )

                for index, factor in enumerate(
                    risk_factors,
                    start=1
                ):

                    feature_name = factor.get(
                        "readable_feature",
                        factor.get("feature", "Shipment detail")
                    )

                    with st.expander(
                        f"{index}. {feature_name}"
                    ):

                        input_value = factor.get("input_value")

                        if input_value is not None:
                            st.write(
                                f"**Recorded input value:** {input_value}"
                            )

                        st.write(
                            "**What this means:** "
                            + factor.get(
                                "explanation",
                                "This recorded value influenced the model prediction."
                            )
                        )

                        st.write(
                            "**What to check:** "
                            + factor.get(
                                "suggestion",
                                "Review the shipment's tracking and operational records."
                            )
                        )

            else:

                st.info(
                    "The model did not return any original input features "
                    "pushing its prediction toward higher risk."
                )

     
            # RISK-REDUCING FACTORS
 

            st.header("✅ Factors Pushing Risk Lower")

            protective_factors = result.get(
                "protective_factors",
                []
            )

            if protective_factors:

                st.caption(
                    "These recorded input values were associated by "
                    "the model with lower late-delivery risk. They do "
                    "not guarantee on-time delivery."
                )

                for index, factor in enumerate(
                    protective_factors,
                    start=1
                ):

                    feature_name = factor.get(
                        "readable_feature",
                        factor.get("feature", "Shipment detail")
                    )

                    with st.expander(
                        f"{index}. {feature_name}"
                    ):

                        input_value = factor.get("input_value")

                        if input_value is not None:
                            st.write(
                                f"**Recorded input value:** {input_value}"
                            )

                        st.write(
                            "**What this means:** "
                            + factor.get(
                                "explanation",
                                "This recorded value pushed the model's "
                                "prediction toward lower risk."
                            )
                        )

                        st.write(
                            "**What to check:** "
                            + factor.get(
                                "suggestion",
                                "Continue monitoring the shipment's progress."
                            )
                        )

            else:

                st.info(
                    "The model did not return any original input features "
                    "pushing its prediction toward lower risk."
                )

      
            # RECOMMENDED ACTIONS
        
            st.header("💡 Recommended Actions for This Shipment")

            recommendations = result.get(
                "recommendations",
                []
            )

            if recommendations:

                for index, recommendation in enumerate(
                    recommendations,
                    start=1
                ):

                    st.write(
                        f"{index}. {recommendation}"
                    )

            else:

                st.info(
                    "No recommendations were returned by the API."
                )

           
            # PLAIN-ENGLISH INTERPRETATION NOTE
           

            with st.expander("ℹ️ Understanding this prediction"):

                st.write(
                    result.get(
                        "explanation_note",
                        "The prediction is an estimate based on patterns "
                        "learned from historical shipment data."
                    )
                )

                st.write(
                    "**Remember:** A high-risk prediction does not mean "
                    "a shipment will definitely be late, and a low-risk "
                    "prediction does not guarantee on-time delivery."
                )

                st.write(
                    "Use the explanations to decide what to investigate. "
                    "Verify recommendations against current tracking "
                    "information and actual operational records."
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "Cannot connect to FastAPI. Start the backend in another "
                "terminal with: python -m uvicorn api.main:app --reload"
            )

        except requests.exceptions.HTTPError as error:

            st.error(f"API returned an error: {error}")

            if response is not None:
                st.code(response.text)

        except requests.exceptions.RequestException as error:

            st.error(f"Request failed: {error}")

        except (KeyError, ValueError, TypeError) as error:

            st.error(
                "The prediction response could not be interpreted. "
                f"Details: {error}"
            )
