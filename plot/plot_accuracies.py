from matplotlib import pyplot as plt

from .config import *
from .set_plot_format import format_axes_and_grid, select_charts_font


def plot_accuracies_over_weeks(weeks_train_val, weeks_test, df_results_overview_2022, val_column, test_column, title, filename="val_accuracies_2022.svg", palette=KUL_PALETTE, font=FONT_KULEUVEN):
    val_accuracies_retrain_week_by_week = get_list_accuracies_element_df(
        df_results_overview_2022,
        'retrain_week_by_week',
        val_column
    )
    test_accuracies_retrain_week_by_week = get_list_accuracies_element_df(
        df_results_overview_2022,
        'retrain_week_by_week',
        test_column
    )

    val_accuracies_test_on_trained_model_20_21 = get_list_accuracies_element_df(
        df_results_overview_2022,
        'test_on_trained_model_20_21',
        val_column
    )
    test_accuracies_test_on_trained_model_20_21 = get_list_accuracies_element_df(
        df_results_overview_2022,
        'test_on_trained_model_20_21',
        test_column
    )

    select_charts_font(font)
    digit_weeks = [int(''.join(filter(str.isdigit, week)))
                   for week in weeks_train_val]

    fig, ax = plt.subplots(figsize=(18, 8))

    ax.plot(
        digit_weeks,
        val_accuracies_retrain_week_by_week,
        label="Validation (retrain week by week)",
        color=palette['first_color']
    )
    ax.plot(
        digit_weeks,
        test_accuracies_retrain_week_by_week,
        label=f"Test (retrain week by week) with final weeks: {weeks_test}",
        color=palette['second_color']
    )

    ax.plot(
        digit_weeks,
        val_accuracies_test_on_trained_model_20_21,
        label="Validation (over model from 2020-2021)",
        color=palette['first_color'],
        linestyle='dotted'
    )

    ax.plot(
        digit_weeks,
        test_accuracies_test_on_trained_model_20_21,
        label=f"Test (over model from 2020-2021) with final weeks: {weeks_test}",
        color=palette['second_color'],
        linestyle='dotted'
    )

    ax.set_xticks(digit_weeks)
    ax.set_title(title, fontsize=16, color=palette['text_color'])
    ax.set_xlabel('Weeks', fontsize=12, color=palette['text_color'])
    ax.set_ylabel('Accuracies (%)', fontsize=12, color=palette['text_color'])
    format_axes_and_grid(ax, palette)
    plt.legend(frameon=False, labelcolor=palette['text_color'], fontsize=12)
    fig.savefig(filename)


def get_list_accuracies_element_df(df, experiment, column):
    return df.loc[experiment, column]
