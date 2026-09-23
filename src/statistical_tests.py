"""Bivariate significance tests: which features are associated with
campaign acceptance? Ported from the R chi-square/ANOVA section of
`Codes.R`.
"""
import pandas as pd
from scipy import stats


def chi_square_test(df: pd.DataFrame, categorical_col: str, target_col: str = "Accepted_Cmp"):
    contingency = pd.crosstab(df[categorical_col], df[target_col])
    chi2, p, dof, _ = stats.chi2_contingency(contingency)
    return {"variable": categorical_col, "test": "chi-square", "statistic": chi2, "p_value": p}


def anova_test(df: pd.DataFrame, numeric_col: str, target_col: str = "Accepted_Cmp"):
    groups = [group[numeric_col].values for _, group in df.groupby(target_col, observed=True)]
    f_stat, p = stats.f_oneway(*groups)
    return {"variable": numeric_col, "test": "one-way ANOVA", "statistic": f_stat, "p_value": p}


def run_all_bivariate_tests(df: pd.DataFrame) -> pd.DataFrame:
    categorical_vars = ["Marital_Status", "children", "Education", "Complain"]
    numeric_vars = [
        "age", "Income", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",
        "NumWebVisitsMonth", "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts",
        "MntSweetProducts", "MntGoldProds", "duration",
    ]

    results = [chi_square_test(df, col) for col in categorical_vars]
    results += [anova_test(df, col) for col in numeric_vars]

    results_df = pd.DataFrame(results)
    results_df["significant_at_0.05"] = results_df["p_value"] < 0.05
    return results_df.sort_values("p_value")


if __name__ == "__main__":
    from data_processing import load_and_clean_data

    df = load_and_clean_data()
    results = run_all_bivariate_tests(df)
    pd.set_option("display.max_rows", None)
    print(results.to_string(index=False))
