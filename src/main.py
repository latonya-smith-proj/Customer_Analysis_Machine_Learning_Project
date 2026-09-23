"""End-to-end pipeline: clean data -> bivariate tests -> model
comparison -> best-model evaluation -> cluster analysis of customers
who accepted a promotion."""
from clustering import cluster_accepted_customers, plot_elbow, profile_plot
from data_processing import load_and_clean_data
from model_comparison import compare_models, evaluate_best_model, feature_importance
from statistical_tests import run_all_bivariate_tests


def main():
    print("=" * 70)
    print("STEP 1: Load and clean data")
    print("=" * 70)
    df = load_and_clean_data()
    print(f"Cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns\n")

    print("=" * 70)
    print("STEP 2: Bivariate significance tests vs. campaign acceptance")
    print("=" * 70)
    test_results = run_all_bivariate_tests(df)
    print(test_results.to_string(index=False), "\n")

    print("=" * 70)
    print("STEP 3: Compare classifiers (10-fold CV, ROC AUC)")
    print("=" * 70)
    comparison = compare_models(df)
    print("\n", comparison.to_string(index=False), "\n")

    print("=" * 70)
    print("STEP 4: Evaluate the best model on a held-out test set")
    print("=" * 70)
    best_model_name = comparison.iloc[0]["model"]
    pipeline, X_test, y_test, probs = evaluate_best_model(df, model_name=best_model_name)
    if hasattr(pipeline.named_steps["model"], "feature_importances_"):
        print("\nTop features:\n", feature_importance(pipeline).to_string(index=False))

    print("\n" + "=" * 70)
    print("STEP 5: Cluster customers who accepted a promotion")
    print("=" * 70)
    clustered, km, scaled = cluster_accepted_customers(df)
    plot_elbow(scaled)
    profile_plot(clustered)
    print(clustered["cluster"].value_counts().sort_index())


if __name__ == "__main__":
    main()
