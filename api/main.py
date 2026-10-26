
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI
from pydantic import BaseModel, Field


# 1. LOAD MODEL AND PREPROCESSOR


PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "xgboost_supply_chain_model.joblib"
)

artifacts = joblib.load(MODEL_PATH)

model = artifacts["model"]
preprocessor = artifacts["preprocessor"]
threshold = float(artifacts["threshold"])

feature_names = list(preprocessor.get_feature_names_out())

explainer = shap.TreeExplainer(model)



# 2. FASTAPI APPLICATION


app = FastAPI(
    title="Smart Supply Chain Risk Predictor",
    description=(
        "Predict late-delivery risk, explain individual predictions "
        "using SHAP, and generate shipment-specific investigation actions."
    ),
    version="3.0.0"
)



# 3. INPUT SCHEMA


class ShipmentInput(BaseModel):
    days_for_shipment_scheduled: int = Field(ge=0)
    order_item_discount: float = Field(ge=0)
    order_item_quantity: int = Field(ge=1)
    sales: float = Field(ge=0)
    product_price: float = Field(ge=0)

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

    order_year: int = Field(ge=1900, le=2100)
    order_month: int = Field(ge=1, le=12)
    order_day: int = Field(ge=1, le=31)
    order_day_of_week: int = Field(ge=0, le=6)
    order_hour: int = Field(ge=0, le=23)

    type: str
    customer_country: str
    customer_segment: str
    market: str
    customer_state: str
    order_region: str
    order_country: str

    category_id: int = Field(ge=0)
    product_card_id: int = Field(ge=0)
    department_id: int = Field(ge=0)



# 4. FEATURE DEFINITIONS


# These are the original features supplied to the model.
# SHAP explanations are grouped back to these fields after
# preprocessing and one-hot encoding.

FEATURE_LABELS = {
    "Days for shipment (scheduled)": "Scheduled shipment days",
    "Order Item Discount": "Order item discount",
    "Order Item Quantity": "Order item quantity",
    "Sales": "Sales value",
    "Product Price": "Product price",
    "Latitude": "Delivery latitude",
    "Longitude": "Delivery longitude",
    "Order Year": "Order year",
    "Order Month": "Order month",
    "Order Day": "Order day",
    "Order Day of Week": "Order weekday",
    "Order Hour": "Order hour",
    "Type": "Shipment type",
    "Customer Country": "Customer country",
    "Customer Segment": "Customer segment",
    "Market": "Market",
    "Customer State": "Customer state",
    "Order Region": "Order region",
    "Order Country": "Order country",
    "Category Id": "Category ID",
    "Product Card Id": "Product card ID",
    "Department Id": "Department ID",
}

CATEGORICAL_COLUMNS = [
    "Type",
    "Customer Country",
    "Customer Segment",
    "Market",
    "Customer State",
    "Order Region",
    "Order Country",
    "Category Id",
    "Product Card Id",
    "Department Id",
]



# 5. FEATURE MAPPING HELPERS


def clean_transformed_feature_name(feature_name):
    """Remove the ColumnTransformer prefix from a feature name."""

    feature = str(feature_name)

    for prefix in ("num__", "cat__"):
        if feature.startswith(prefix):
            feature = feature[len(prefix):]

    return feature


def find_original_feature(transformed_feature):
    """
    Map a transformed feature to its original input column.

    Examples:
        num__Sales -> Sales
        cat__Type_SHIPPING -> Type
        cat__Order Region_South Asia -> Order Region
    """

    cleaned = clean_transformed_feature_name(transformed_feature)

    # Exact matches handle numerical features and any unencoded fields.
    if cleaned in FEATURE_LABELS:
        return cleaned

    # Longest-first matching avoids ambiguous column-name matches.
    for column in sorted(FEATURE_LABELS, key=len, reverse=True):
        if cleaned.startswith(column + "_"):
            return column

    return cleaned


def format_value(value):
    """Format the actual input value for plain-English explanations."""

    if isinstance(value, (np.integer, int)):
        return str(int(value))

    if isinstance(value, (np.floating, float)):
        return f"{float(value):,.4f}".rstrip("0").rstrip(".")

    return str(value)


def build_input_data(shipment):
    """Map API field names to the exact original model column names."""

    return {
        "Days for shipment (scheduled)":
            shipment.days_for_shipment_scheduled,
        "Order Item Discount": shipment.order_item_discount,
        "Order Item Quantity": shipment.order_item_quantity,
        "Sales": shipment.sales,
        "Product Price": shipment.product_price,
        "Latitude": shipment.latitude,
        "Longitude": shipment.longitude,
        "Order Year": shipment.order_year,
        "Order Month": shipment.order_month,
        "Order Day": shipment.order_day,
        "Order Day of Week": shipment.order_day_of_week,
        "Order Hour": shipment.order_hour,
        "Type": shipment.type,
        "Customer Country": shipment.customer_country,
        "Customer Segment": shipment.customer_segment,
        "Market": shipment.market,
        "Customer State": shipment.customer_state,
        "Order Region": shipment.order_region,
        "Order Country": shipment.order_country,
        "Category Id": shipment.category_id,
        "Product Card Id": shipment.product_card_id,
        "Department Id": shipment.department_id,
    }


# 6. SHAP EXPLANATION HELPERS


def calculate_shap_contributions(processed_data):
    """Calculate SHAP contributions for the supplied shipment."""

    processed_dense = (
        processed_data.toarray()
        if hasattr(processed_data, "toarray")
        else np.asarray(processed_data)
    )

    shap_values = explainer.shap_values(processed_dense)

    # Handle common SHAP output formats.
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            shap_values = shap_values[1]
        else:
            shap_values = shap_values[0]

    shap_array = np.asarray(shap_values)

    # Some binary classifiers return:
    # (rows, features, classes).
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 1]

    contributions = shap_array.reshape(-1)

    if len(contributions) != len(feature_names):
        raise ValueError(
            "The number of SHAP contributions does not match "
            "the preprocessor's transformed feature names."
        )

    transformed_df = pd.DataFrame({
        "Transformed_Feature": feature_names,
        "SHAP_Contribution": contributions,
    })

    transformed_df["Original_Feature"] = (
        transformed_df["Transformed_Feature"]
        .map(find_original_feature)
    )

    # Group one-hot encoded columns back into their original fields.
    grouped_df = (
        transformed_df
        .groupby("Original_Feature", as_index=False)
        ["SHAP_Contribution"]
        .sum()
    )

    return grouped_df


def explain_feature(feature_name, actual_value, contribution):
    """
    Explain the actual input value and the direction of its
    influence on the model prediction.

    SHAP is an explanation of model behavior, not proof of causation.
    """

    label = FEATURE_LABELS.get(feature_name, feature_name)
    value_text = format_value(actual_value)

    if contribution > 0:
        direction = "higher"
        effect = (
            "For this prediction, the model associated this recorded "
            "value with higher late-delivery risk."
        )
    elif contribution < 0:
        direction = "lower"
        effect = (
            "For this prediction, the model associated this recorded "
            "value with lower late-delivery risk."
        )
    else:
        direction = "neutral"
        effect = (
            "For this prediction, this feature had no net SHAP "
            "contribution after grouping its transformed columns."
        )

    if feature_name == "Days for shipment (scheduled)":
        detail = (
            f"The scheduled shipment duration entered for this order "
            f"is {value_text} day(s)."
        )

    elif feature_name == "Order Item Quantity":
        detail = (
            f"The order contains {value_text} item(s)."
        )

    elif feature_name == "Order Item Discount":
        detail = (
            f"The recorded order-item discount is {value_text}."
        )

    elif feature_name == "Sales":
        detail = (
            f"The recorded sales value is {value_text}."
        )

    elif feature_name == "Product Price":
        detail = (
            f"The recorded product price is {value_text}."
        )

    elif feature_name == "Order Hour":
        detail = (
            f"The recorded order hour is {value_text} "
            "on the 24-hour clock."
        )

    elif feature_name == "Order Month":
        detail = (
            f"The recorded order month is {value_text}."
        )

    elif feature_name == "Order Day":
        detail = (
            f"The recorded order day is {value_text}."
        )

    elif feature_name == "Order Year":
        detail = (
            f"The recorded order year is {value_text}."
        )

    elif feature_name == "Order Day of Week":
        weekday_names = [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"
        ]

        weekday = int(actual_value)

        if 0 <= weekday <= 6:
            detail = (
                f"The recorded order weekday is {weekday_names[weekday]} "
                f"(value {weekday})."
            )
        else:
            detail = f"The recorded order weekday value is {value_text}."

    elif feature_name in ("Latitude", "Longitude"):
        detail = (
            f"The recorded {label.lower()} is {value_text}. "
            "This represents the geographic information supplied "
            "to the model."
        )

    elif feature_name in CATEGORICAL_COLUMNS:
        detail = (
            f"The selected {label.lower()} is '{value_text}'."
        )

    else:
        detail = (
            f"The recorded value for {label.lower()} is '{value_text}'."
        )

    return f"{detail} {effect} This is a learned model association, not proof that this value causes a delay."



# 7. SHIPMENT-SPECIFIC INVESTIGATION SUGGESTIONS


def get_feature_suggestion(feature_name, actual_value):
    """
    Return an operational check based on the actual value.

    These are investigation suggestions, not claims that a problem
    has already been confirmed.
    """

    value = format_value(actual_value)

    if feature_name == "Days for shipment (scheduled)":
        return (
            f"Check the planned delivery commitment for the "
            f"{value}-day scheduled shipment. Compare the current "
            "tracking milestones and route transit time with that "
            "commitment."
        )

    if feature_name == "Order Item Quantity":
        quantity = int(actual_value)

        return (
            f"Confirm inventory, picking, and packing for all "
            f"{quantity} ordered item(s). Compare the fulfillment "
            "status with the shipment's planned dispatch time."
        )

    if feature_name == "Order Hour":
        return (
            f"The order hour entered is {value}. Check the order's "
            "actual processing time, warehouse cut-off schedule, "
            "and carrier collection time."
        )

    if feature_name == "Order Month":
        return (
            f"The order month entered is {value}. Compare current "
            "workload and dispatch performance with historical "
            "patterns for the same period."
        )

    if feature_name == "Order Day of Week":
        return (
            "Compare the order date with warehouse operating hours, "
            "dispatch schedules, and carrier collection days."
        )

    if feature_name == "Order Day":
        return (
            f"Check the order date against the actual processing "
            f"and dispatch milestones for day {value} of the month."
        )

    if feature_name == "Order Year":
        return (
            f"Review historical delivery performance and operational "
            f"changes relevant to order year {value}."
        )

    if feature_name == "Type":
        return (
            f"Review the workflow for shipment type '{value}'. "
            "Verify that its processing, dispatch, and carrier "
            "milestones are being met."
        )

    if feature_name == "Order Region":
        return (
            f"Check carrier coverage, route tracking, and transit "
            f"performance for order region '{value}'."
        )

    if feature_name == "Order Country":
        return (
            f"Verify the route and carrier arrangements for "
            f"destination country '{value}'. Check whether the "
            "shipment requires any applicable cross-border processing."
        )

    if feature_name == "Customer Country":
        return (
            f"Verify the destination details for customer country "
            f"'{value}' and check the applicable carrier and transit "
            "arrangements."
        )

    if feature_name == "Customer State":
        return (
            f"Check delivery coverage and recent tracking milestones "
            f"for customer state '{value}'."
        )

    if feature_name == "Latitude":
        return (
            f"Verify the recorded latitude ({value}) against the "
            "intended delivery location and confirm route coverage."
        )

    if feature_name == "Longitude":
        return (
            f"Verify the recorded longitude ({value}) against the "
            "intended delivery location and confirm route coverage."
        )

    if feature_name == "Market":
        return (
            f"Review fulfillment and delivery performance for market "
            f"'{value}' against the shipment's actual tracking status."
        )

    if feature_name == "Customer Segment":
        return (
            f"Review the order's fulfillment workflow for customer "
            f"segment '{value}' and confirm the shipment is progressing "
            "according to its planned schedule."
        )

    if feature_name == "Category Id":
        return (
            f"Check the handling and fulfillment requirements recorded "
            f"for category ID {value}, including any applicable "
            "packing or inventory checks."
        )

    if feature_name == "Product Card Id":
        return (
            f"Verify product ID {value} against the order and check "
            "the associated inventory, picking, and packing status."
        )

    if feature_name == "Department Id":
        return (
            f"Review fulfillment status for department ID {value} "
            "and verify that the order has completed the required "
            "processing and dispatch steps."
        )

    if feature_name == "Order Item Discount":
        return (
            f"Verify the recorded discount ({value}) against the order "
            "details. Review fulfillment status without assuming that "
            "the discount caused a delay."
        )

    if feature_name == "Sales":
        return (
            f"Verify the recorded sales value ({value}) against the "
            "order record. Use actual fulfillment and tracking evidence "
            "to decide whether further action is needed."
        )

    if feature_name == "Product Price":
        return (
            f"Verify the recorded product price ({value}) against the "
            "product record and check any relevant handling requirements."
        )

    return (
        f"Verify the recorded value '{value}' for {feature_name} "
        "and compare it with the original order and shipment records."
    )


def get_protective_suggestion(feature_name, actual_value):
    """Suggest a check that preserves existing positive conditions."""

    value = format_value(actual_value)

    if feature_name == "Days for shipment (scheduled)":
        return (
            f"Keep monitoring progress against the scheduled "
            f"{value}-day shipment window and investigate any missed "
            "delivery milestone."
        )

    if feature_name == "Order Item Quantity":
        quantity = int(actual_value)

        return (
            f"Confirm that all {quantity} item(s) remain accounted "
            "for and that picking, packing, and dispatch are on schedule."
        )

    if feature_name in ("Order Region", "Order Country", "Customer Country"):
        return (
            f"Continue tracking the shipment to '{value}' and verify "
            "that the planned carrier and route remain valid."
        )

    if feature_name == "Type":
        return (
            f"Continue following the workflow for shipment type "
            f"'{value}' and monitor the planned dispatch milestones."
        )

    if feature_name in ("Order Hour", "Order Day", "Order Month",
                        "Order Year", "Order Day of Week"):
        return (
            "Continue checking the planned dispatch schedule and "
            "monitor for any missed warehouse or carrier milestones."
        )

    if feature_name in ("Latitude", "Longitude"):
        return (
            f"Keep the recorded coordinate ({value}) consistent with "
            "the intended delivery address and route."
        )

    return (
        f"Continue routine monitoring and verify that the recorded "
        f"{FEATURE_LABELS.get(feature_name, feature_name).lower()} "
        "remains consistent with the order record."
    )


# 8. HEALTH CHECK


@app.get("/")
def home():
    return {
        "message": "Smart Supply Chain Risk Predictor API is running",
        "model": type(model).__name__,
        "threshold": threshold,
        "explanations": "Grouped, plain-English SHAP explanations",
    }



# 9. SHIPMENT PREDICTION


@app.post("/predict")
def predict_shipment(shipment: ShipmentInput):

    # Actual values entered by the user.
    input_data = build_input_data(shipment)

    input_df = pd.DataFrame([input_data])

    # Preserve the exact feature order used during model training.
    input_df = input_df[preprocessor.feature_names_in_]

    # Apply the original fitted preprocessing pipeline.
    processed_data = preprocessor.transform(input_df)

    # Predict probability of the positive class.
    probability = float(
        model.predict_proba(processed_data)[0, 1]
    )

    predicted_late_risk = int(probability >= threshold)

    # Apply the project's existing business risk categories.
    if probability >= 0.70:
        risk_category = "High Risk"
    elif probability >= threshold:
        risk_category = "Medium Risk"
    else:
        risk_category = "Low Risk"

    # Group transformed SHAP values into original input features.
    grouped_df = calculate_shap_contributions(processed_data)

    grouped_df["Actual_Value"] = grouped_df["Original_Feature"].map(
        input_data
    )

    # Only explain original fields that exist in the actual input.
    grouped_df = grouped_df[
        grouped_df["Original_Feature"].isin(input_data.keys())
    ].copy()

    # Select original features pushing model output upward.
    positive_factors = (
        grouped_df[grouped_df["SHAP_Contribution"] > 0]
        .sort_values("SHAP_Contribution", ascending=False)
        .head(5)
    )

    # Select original features pushing model output downward.
    negative_factors = (
        grouped_df[grouped_df["SHAP_Contribution"] < 0]
        .sort_values("SHAP_Contribution", ascending=True)
        .head(3)
    )

    # Build user-friendly risk factors.
    risk_factors = []

    for _, row in positive_factors.iterrows():
        feature = row["Original_Feature"]
        actual_value = row["Actual_Value"]
        contribution = float(row["SHAP_Contribution"])

        risk_factors.append({
            "feature": feature,
            "readable_feature": FEATURE_LABELS.get(feature, feature),
            "input_value": format_value(actual_value),
            "explanation": explain_feature(
                feature, actual_value, contribution
            ),
            "suggestion": get_feature_suggestion(
                feature, actual_value
            ),
        })

    # Build user-friendly risk-reducing factors.
    protective_factors = []

    for _, row in negative_factors.iterrows():
        feature = row["Original_Feature"]
        actual_value = row["Actual_Value"]
        contribution = float(row["SHAP_Contribution"])

        protective_factors.append({
            "feature": feature,
            "readable_feature": FEATURE_LABELS.get(feature, feature),
            "input_value": format_value(actual_value),
            "explanation": explain_feature(
                feature, actual_value, contribution
            ),
            "suggestion": get_protective_suggestion(
                feature, actual_value
            ),
        })

    # General actions based on the predicted risk category.
    if risk_category == "High Risk":
        recommended_actions = [
            "Prioritize this shipment for an operational review.",
            "Check actual tracking events, inventory readiness, and dispatch status.",
            "Compare the current tracking milestones with the planned delivery commitment.",
            "Contact the carrier if tracking evidence indicates a missed or threatened milestone.",
            "Notify the customer if verified shipment information supports a possible delay.",
        ]

    elif risk_category == "Medium Risk":
        recommended_actions = [
            "Monitor the shipment's tracking updates and planned dispatch milestones.",
            "Review the strongest model risk indicators against actual shipment records.",
            "Escalate if a milestone is missed or new tracking evidence indicates a problem.",
        ]

    else:
        recommended_actions = [
            "Continue routine shipment monitoring.",
            "Verify that dispatch and delivery milestones remain on schedule.",
            "Reassess if tracking or operational information indicates a new risk.",
        ]

    # Add checks associated with the strongest model risk indicators.
    for factor in risk_factors[:3]:
        recommendation = factor["suggestion"]

        if recommendation not in recommended_actions:
            recommended_actions.append(recommendation)

    return {
        "late_delivery_probability": round(probability, 4),
        "predicted_late_risk": predicted_late_risk,
        "risk_category": risk_category,
        "classification_threshold": threshold,
        "message": (
            "Shipment flagged for priority review"
            if predicted_late_risk
            else "Shipment is below the selected classification threshold"
        ),
        "risk_factors": risk_factors,
        "protective_factors": protective_factors,
        "recommendations": recommended_actions,
        "explanation_note": (
            "The probability is the model's estimate of late delivery "
            "based on patterns learned from historical shipment data. "
            "SHAP explanations show how the recorded input values "
            "influenced this particular prediction. One-hot encoded "
            "features are grouped into their original input fields. "
            "These associations do not prove real-world causes. "
            "Recommendations are checks to perform, not confirmed "
            "operational problems. Verify them using current tracking, "
            "carrier, warehouse, and order records."
        ),
    }
