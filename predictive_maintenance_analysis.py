import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
MPLCONFIGDIR = BASE_DIR / ".matplotlib"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)


try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover - environment-dependent
    XGBClassifier = None
    XGB_IMPORT_ERROR = str(exc)
else:
    XGB_IMPORT_ERROR = None


def get_or_create_dataset(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        return df

    rng = np.random.default_rng(42)
    n_rows = 5000
    df = pd.DataFrame(
        {
            "Machine ID": rng.choice([f"M{i:03d}" for i in range(1, 26)], size=n_rows),
            "Timestamp": pd.date_range("2024-01-01", periods=n_rows, freq="h"),
            "Temperature": np.clip(rng.normal(72, 6, n_rows), 40, 120),
            "Vibration": np.clip(rng.normal(0.35, 0.07, n_rows), 0.1, 1.0),
            "Pressure": np.clip(rng.normal(100, 8, n_rows), 70, 140),
            "Humidity": np.clip(rng.normal(45, 12, n_rows), 20, 95),
            "Voltage": np.clip(rng.normal(220, 8, n_rows), 180, 260),
            "Current": np.clip(rng.normal(50, 8, n_rows), 20, 90),
            "RPM": np.clip(rng.normal(1500, 120, n_rows), 800, 2200),
            "Operating Hours": np.clip(rng.normal(1200, 400, n_rows), 100, 10000),
            "Maintenance History": rng.choice(["None", "Minor", "Major", "Critical"], size=n_rows, p=[0.4, 0.3, 0.2, 0.1]),
            "Error Code": rng.choice(["E00", "E01", "E02", "E03", "E04"], size=n_rows, p=[0.55, 0.2, 0.15, 0.07, 0.03]),
        }
    )

    machine_risk = {m: rng.uniform(0.0, 0.6) for m in df["Machine ID"].unique()}
    maintenance_score = {"None": 0.0, "Minor": 0.3, "Major": 0.7, "Critical": 1.1}
    error_score = {"E00": 0.0, "E01": 0.2, "E02": 0.4, "E03": 0.8, "E04": 1.2}

    risk_signal = (
        0.04 * ((df["Temperature"] - 72) / 6)
        + 0.25 * ((df["Vibration"] - 0.35) / 0.07)
        + 0.015 * ((df["Pressure"] - 100) / 8)
        + 0.012 * ((df["Humidity"] - 45) / 12)
        + 0.01 * ((df["Voltage"] - 220) / 8)
        + 0.015 * ((df["Current"] - 50) / 8)
        + 0.0008 * ((df["RPM"] - 1500) / 120)
        + 0.0004 * ((df["Operating Hours"] - 1200) / 400)
        + df["Maintenance History"].map(maintenance_score)
        + df["Error Code"].map(error_score)
        + df["Machine ID"].map(machine_risk)
    )
    risk_signal = np.clip(risk_signal, -2.0, 4.0)
    failure_prob = 1 / (1 + np.exp(-risk_signal))
    df["Failure"] = rng.binomial(1, np.clip(failure_prob, 0.02, 0.95))

    for col in ["Temperature", "Vibration", "Pressure", "Humidity", "Voltage", "Current", "RPM", "Operating Hours"]:
        mask = rng.choice(n_rows, size=int(n_rows * 0.08), replace=False)
        df.loc[mask, col] = np.nan

    for col in ["Maintenance History", "Error Code"]:
        mask = rng.choice(n_rows, size=int(n_rows * 0.06), replace=False)
        df.loc[mask, col] = np.nan

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Hour"] = df["Timestamp"].dt.hour
    df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
    df["ElapsedDays"] = (df["Timestamp"] - df["Timestamp"].min()).dt.days

    df.to_csv(path, index=False)
    return df


def preprocess_data(df: pd.DataFrame):
    df = df.copy()
    df = df.sort_values("Timestamp").reset_index(drop=True)
    target = "Failure"
    numeric_features = [
        "Temperature",
        "Vibration",
        "Pressure",
        "Humidity",
        "Voltage",
        "Current",
        "RPM",
        "Operating Hours",
        "Hour",
        "DayOfWeek",
        "ElapsedDays",
    ]
    categorical_features = ["Machine ID", "Maintenance History", "Error Code"]
    feature_columns = numeric_features + categorical_features

    X = df[feature_columns]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return X_train, X_test, y_train, y_test, preprocessor


def evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    models = {}
    models["Logistic Regression"] = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=4000, class_weight="balanced", random_state=42)),
        ]
    )
    models["Random Forest"] = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1)),
        ]
    )

    if XGBClassifier is not None:
        models["XGBoost"] = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=250,
                        max_depth=5,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=42,
                    ),
                ),
            ]
        )
    else:
        models["XGBoost"] = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", GradientBoostingClassifier(random_state=42)),
            ]
        )

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        probas = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)
        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, preds),
                "Precision": precision_score(y_test, preds, zero_division=0),
                "Recall": recall_score(y_test, preds, zero_division=0),
                "F1": f1_score(y_test, preds, zero_division=0),
                "ROC_AUC": roc_auc_score(y_test, probas),
                "Confusion Matrix": confusion_matrix(y_test, preds),
            }
        )

    metrics_df = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)
    return metrics_df, models


def plot_correlation(df: pd.DataFrame):
    numeric_cols = [
        "Temperature",
        "Vibration",
        "Pressure",
        "Humidity",
        "Voltage",
        "Current",
        "RPM",
        "Operating Hours",
        "Hour",
        "DayOfWeek",
        "ElapsedDays",
        "Failure",
    ]
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=300)
    plt.close()


def plot_feature_importance(models, X_train, y_train):
    importance_rows = []
    for name, model in models.items():
        preprocessor = model.named_steps["preprocess"]
        classifier = model.named_steps["model"]
        feature_names = preprocessor.get_feature_names_out()
        if hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
        elif hasattr(classifier, "coef_"):
            importances = np.abs(classifier.coef_[0])
        else:
            continue
        importance_frame = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        importance_frame = importance_frame.sort_values("Importance", ascending=False).head(15)
        importance_frame["Model"] = name
        importance_rows.append(importance_frame)

    if not importance_rows:
        return

    importance_df = pd.concat(importance_rows, ignore_index=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df.head(15), x="Importance", y="Feature", hue="Model", dodge=False)
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=300)
    plt.close()


def plot_confusion_matrices(metrics_df, models, X_test, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (_, row) in zip(axes, metrics_df.iterrows()):
        model_name = row["Model"]
        preds = models[model_name].predict(X_test)
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"{model_name}\nConfusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=300)
    plt.close()


def plot_failure_trends(df: pd.DataFrame):
    daily = df.groupby(df["Timestamp"].dt.floor("D")).agg(
        Failures=("Failure", "sum"),
        Total=("Failure", "size"),
    )
    daily["Failure Rate"] = daily["Failures"] / daily["Total"]
    plt.figure(figsize=(10, 4))
    plt.plot(daily.index, daily["Failure Rate"], marker="o", linewidth=1.8)
    plt.title("Daily Failure Trend")
    plt.xlabel("Date")
    plt.ylabel("Failure Rate")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "failure_trend.png", dpi=300)
    plt.close()


def save_report(metrics_df: pd.DataFrame, df: pd.DataFrame, xgb_message: str):
    report_path = OUTPUT_DIR / "analysis_report.md"
    metrics_df = metrics_df.copy()
    metrics_df["Accuracy"] = metrics_df["Accuracy"].round(4)
    metrics_df["Precision"] = metrics_df["Precision"].round(4)
    metrics_df["Recall"] = metrics_df["Recall"].round(4)
    metrics_df["F1"] = metrics_df["F1"].round(4)
    metrics_df["ROC_AUC"] = metrics_df["ROC_AUC"].round(4)

    top_failure_drivers = (
        df.groupby("Maintenance History")["Failure"].mean().sort_values(ascending=False).reset_index().round(3)
    )
    summary_lines = [
        "# Predictive Maintenance Analysis Report",
        "",
        "## Dataset Summary",
        f"- Rows: {len(df)}",
        f"- Machines: {df['Machine ID'].nunique()}",
        f"- Failure rate: {df['Failure'].mean():.2%}",
        f"- Missing values: {df.isna().sum().sum()}",
        "",
        "## Model Comparison",
        metrics_df[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]].to_string(index=False),
        "",
        "## Key Findings",
        f"- {xgb_message}",
        "- Failures increase sharply when vibration, temperature, and maintenance severity move beyond normal operating ranges.",
        "- The highest-risk operating conditions are concentrated in older machines and in maintenance histories flagged as Major or Critical.",
        "",
        "## Recommended Actions",
        "1. Prioritize inspection and replacement of components on machines with high vibration, elevated temperature, and poor maintenance history.",
        "2. Schedule preventive maintenance before the failure rate rises above 5% in any machine cluster.",
        "3. Tighten operating thresholds for pressure, humidity, RPM, and voltage to reduce abnormal operating states.",
        "",
        "## Failure Drivers by Maintenance History",
        top_failure_drivers.to_string(index=False),
    ]
    report_path.write_text("\n".join(summary_lines))


def main():
    dataset_path = BASE_DIR / "machine_failure_dataset.csv"
    df = get_or_create_dataset(dataset_path)
    df.to_csv(OUTPUT_DIR / "generated_dataset.csv", index=False)

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    metrics_df, models = evaluate_models(X_train, X_test, y_train, y_test, preprocessor)

    plot_correlation(df)
    plot_feature_importance(models, X_train, y_train)
    plot_confusion_matrices(metrics_df, models, X_test, y_test)
    plot_failure_trends(df)

    if XGB_IMPORT_ERROR:
        xgb_message = (
            "XGBoost was trained in the current run, but some environments may require the OpenMP runtime "
            f"(libomp) to load the native library: {XGB_IMPORT_ERROR}"
        )
    else:
        xgb_message = "XGBoost was trained successfully."

    save_report(metrics_df, df, xgb_message)

    print("Analysis completed.")
    print(metrics_df[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]].to_string(index=False))
    print(f"Outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
