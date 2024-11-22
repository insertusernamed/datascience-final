import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    KFold,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import gc
from flask import Flask, render_template, request, jsonify
import requests
from datetime import datetime

# global feature list
FEATURE_NAMES = ["km_driven", "fuel", "seller_type", "transmission", "owner", "car_age"]

df = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")


def preprocess_data(df):
    # prep initial data
    data = df.copy()
    data = data.dropna()

    # calculate car age
    current_year = 2023
    data["car_age"] = current_year - data["year"]

    # convert owner to numeric
    data["owner"] = data["owner"].str.extract("(\d+)").fillna(1).astype(int)

    data = data.drop(["name", "year"], axis=1)

    # encode categorical data
    le = LabelEncoder()
    data["fuel"] = le.fit_transform(data["fuel"])
    data["seller_type"] = le.fit_transform(data["seller_type"])
    data["transmission"] = le.fit_transform(data["transmission"])

    joblib.dump(le, "label_encoder.joblib")
    return data


processed_data = preprocess_data(df)

feature_names = ["km_driven", "fuel", "seller_type", "transmission", "owner", "car_age"]

# split data
X = processed_data.drop("selling_price", axis=1)
y = processed_data["selling_price"]

# scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=FEATURE_NAMES)
joblib.dump(scaler, "scaler.joblib")

# add polynomial features
poly = PolynomialFeatures(degree=1, include_bias=False)
X_poly = poly.fit_transform(X_scaled)
poly_feature_names = [f"poly_{i}" for i in range(X_poly.shape[1])]
X_poly = pd.DataFrame(X_poly, columns=poly_feature_names)
del X_scaled
gc.collect()

feature_names = [f"feature_{i}" for i in range(X_poly.shape[1])]

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)
del X_poly
gc.collect()

# model parameters
param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [15, 20],
    "min_samples_split": [5],
    "min_samples_leaf": [2],
    "max_features": ["sqrt"],
}

# init model
rf_model = RandomForestRegressor(
    random_state=42,
    n_jobs=-1,
    warm_start=True,
    oob_score=True,
)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=kfold,
    n_jobs=-1,
    verbose=1,
    scoring="r2",
    error_score="raise",
)

grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_

# save model artifacts
joblib.dump(best_rf_model, "best_car_price_model.joblib")
joblib.dump(poly, "poly_features.joblib")

# predictions
y_pred_train = best_rf_model.predict(X_train)
y_pred_test = best_rf_model.predict(X_test)

# calculate metrics
metrics = {
    "train r²": r2_score(y_train, y_pred_train),
    "test r²": r2_score(y_test, y_pred_test),
    "train rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
    "test rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
    "train mae": mean_absolute_error(y_train, y_pred_train),
    "test mae": mean_absolute_error(y_test, y_pred_test),
}

# save metrics
with open("model_performance.txt", "w") as f:
    f.write("model performance metrics:\n")
    for metric, value in metrics.items():
        f.write(f"{metric}: {value:.4f}\n")
    f.write(f"\nbest parameters:\n{grid_search.best_params_}")

print("\nmodel performance metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# feature importance
try:
    feature_importance = pd.DataFrame(
        {"feature": feature_names, "importance": best_rf_model.feature_importances_}
    )
    print("\ntop 10 feature importance:")
    print(feature_importance.nlargest(10, "importance"))
except Exception as e:
    print(f"\nerror calculating feature importance: {str(e)}")

# flask setup
app = Flask(__name__)

# load saved models
model = joblib.load("best_car_price_model.joblib")
scaler = joblib.load("scaler.joblib")
poly = joblib.load("poly_features.joblib")


def convert_inr_to_cad(amount_inr):
    try:
        response = requests.get("https://api.exchangerate-api.com/v4/latest/INR")
        rates = response.json()["rates"]
        cad_rate = rates["CAD"]
        return amount_inr * cad_rate
    except:
        return amount_inr * 0.016


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # get input data
        features = {
            "year": int(request.form["year"]),
            "km_driven": float(request.form["km_driven"]),
            "fuel": request.form["fuel"],
            "seller_type": request.form["seller_type"],
            "transmission": request.form["transmission"],
            "owner": int(request.form["owner"]),
        }

        current_year = 2023
        features["car_age"] = current_year - features["year"]

        input_df = pd.DataFrame(
            [
                [
                    features["km_driven"],
                    features["fuel"],
                    features["seller_type"],
                    features["transmission"],
                    features["owner"],
                    features["car_age"],
                ]
            ],
            columns=FEATURE_NAMES,
        )

        # transform input
        input_scaled = scaler.transform(input_df)
        input_scaled = pd.DataFrame(input_scaled, columns=FEATURE_NAMES)
        input_poly = poly.transform(input_scaled)

        prediction = model.predict(input_poly)[0]
        prediction_cad = convert_inr_to_cad(prediction)

        return render_template(
            "result.html",
            prediction=f"${prediction_cad:,.2f} CAD",
            prediction_inr=f"₹{prediction:,.2f}",
            features=features,
        )
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
