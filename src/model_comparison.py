"""Compare classifiers for predicting campaign acceptance.

Ported from the R `caret`-based model comparison in `Codes.R`:
KNN, logistic regression (plain + L1-regularized), random forest,
decision tree, and gradient boosting, all scored on ROC AUC via
10-fold cross-validation.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 1234


def build_preprocessor(df: pd.DataFrame, target_col: str = "Accepted_Cmp") -> ColumnTransformer:
    feature_cols = [c for c in df.columns if c != target_col]
    categorical_cols = [c for c in feature_cols if df[c].dtype.name in ("category", "object")]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


def get_models() -> dict:
    return {
        "knn": KNeighborsClassifier(),
        "logistic_regression": LogisticRegression(max_iter=2000),
        "logistic_regression_l1": LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def compare_models(df: pd.DataFrame, target_col: str = "Accepted_Cmp") -> pd.DataFrame:
    X = df.drop(columns=[target_col])
    y = (df[target_col] == "Yes").astype(int)

    preprocessor = build_preprocessor(df, target_col)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    results = []
    for name, model in get_models().items():
        pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        results.append({"model": name, "mean_roc_auc": scores.mean(), "std_roc_auc": scores.std()})
        print(f"{name}: mean ROC AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")

    return pd.DataFrame(results).sort_values("mean_roc_auc", ascending=False).reset_index(drop=True)


def evaluate_best_model(
    df: pd.DataFrame,
    target_col: str = "Accepted_Cmp",
    model_name: str = "random_forest",
    threshold: float = 0.19,
    test_size: float = 0.2,
):
    """Train/test split, fit the chosen model, and evaluate at a custom
    decision threshold (mirrors the 0.19 cutoff used in the original R
    analysis, chosen to favor sensitivity)."""
    X = df.drop(columns=[target_col])
    y = (df[target_col] == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(df, target_col)
    model = get_models()[model_name]
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs > threshold).astype(int)

    print(f"\nEvaluation of '{model_name}' on the held-out test set (threshold = {threshold}):\n")
    print(classification_report(y_test, preds, target_names=["No", "Yes"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    print(f"Test ROC AUC: {roc_auc_score(y_test, probs):.4f}")

    return pipeline, X_test, y_test, probs


def feature_importance(pipeline: Pipeline, top_n: int = 15) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        raise ValueError(f"{type(model).__name__} does not expose feature_importances_")

    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importances = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    return importances.head(top_n).reset_index(drop=True)


if __name__ == "__main__":
    from data_processing import load_and_clean_data

    df = load_and_clean_data()
    comparison = compare_models(df)
    print("\nModel comparison (ranked by mean ROC AUC):\n", comparison.to_string(index=False))

    best_model_name = comparison.iloc[0]["model"]
    pipeline, X_test, y_test, probs = evaluate_best_model(df, model_name=best_model_name)

    if hasattr(pipeline.named_steps["model"], "feature_importances_"):
        print("\nTop features:\n", feature_importance(pipeline).to_string(index=False))
