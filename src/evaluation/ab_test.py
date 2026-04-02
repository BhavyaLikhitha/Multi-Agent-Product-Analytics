"""A/B testing: compare base vs fine-tuned model summaries.

Runs a two-sample t-test per metric to determine if the
fine-tuned model is statistically better than the base.
Target: p < 0.05 for at least accuracy and completeness.
"""

import pandas as pd
from loguru import logger
from scipy import stats

RESULTS_PATH = "data/processed/evaluation_results.csv"
CRITERIA = [
    "accuracy",
    "completeness",
    "actionability",
    "conciseness",
    "overall_score",
]


def run_ab_test(
    base_model="groq_base",
    finetuned_model="mistral_finetuned",
):
    """Compare two models using paired t-test."""
    df = pd.read_csv(RESULTS_PATH)

    base = df[df["model_name"] == base_model]
    finetuned = df[df["model_name"] == finetuned_model]

    if len(base) == 0:
        logger.error(f"No results for {base_model}")
        return
    if len(finetuned) == 0:
        logger.error(f"No results for {finetuned_model}")
        return

    # Align on common ASINs
    common = set(base["asin"]) & set(finetuned["asin"])
    if len(common) == 0:
        logger.error("No common ASINs between models")
        return

    base = base[base["asin"].isin(common)].sort_values("asin")
    finetuned = finetuned[
        finetuned["asin"].isin(common)
    ].sort_values("asin")

    logger.info(f"Comparing {len(common)} products")
    logger.info(f"  Base model: {base_model}")
    logger.info(f"  Fine-tuned: {finetuned_model}")
    logger.info("")

    print(
        f"{'Metric':<16} {'Base':>6} {'Tuned':>6} "
        f"{'Diff':>6} {'p-value':>8} {'Sig?':>5}"
    )
    print("-" * 55)

    significant_count = 0

    for metric in CRITERIA:
        base_scores = base[metric].values
        ft_scores = finetuned[metric].values

        base_mean = base_scores.mean()
        ft_mean = ft_scores.mean()
        diff = ft_mean - base_mean

        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(
            ft_scores, base_scores, alternative="greater"
        )

        sig = "YES" if p_value < 0.05 else "no"
        if p_value < 0.05:
            significant_count += 1

        print(
            f"{metric:<16} {base_mean:>6.2f} {ft_mean:>6.2f} "
            f"{diff:>+6.2f} {p_value:>8.4f} {sig:>5}"
        )

    print("-" * 55)
    print(
        f"\nSignificant improvements: "
        f"{significant_count}/{len(CRITERIA)}"
    )

    target = significant_count >= 2
    status = "PASS" if target else "FAIL"
    print(f"Target (>=2 significant): {status}")

    return significant_count


if __name__ == "__main__":
    run_ab_test()
