import matplotlib.pyplot as plt

from .config import *
from .plot_accuracies import get_list_accuracies_element_df
from .set_plot_format import format_axes_and_grid, select_charts_font


def plot_accuracies_critic_insects(insect, weeks_train_val, weeks_test, df_results_overview_2022, val_column, test_column, title, filename="val_accuracies_2022.svg", palette=KUL_PALETTE, font=FONT_KULEUVEN):
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

    val_total_retrain_week_by_week = get_list_accuracies_element_df(
        df_results_overview_2022,
        'test_on_trained_model_20_21',
        'val_total_' + insect
    )

    test_total_retrain_week_by_week = get_list_accuracies_element_df(
        df_results_overview_2022,
        'test_on_trained_model_20_21',
        'test_total_' + insect
    )

    #The validation and test sets were the same for both experiments, so reading this info once is enough
    
    select_charts_font(font)
    digit_weeks = [int(''.join(filter(str.isdigit, week)))
                   for week in weeks_train_val]

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(15, 8))

    plt.suptitle(title, fontsize=16, color=palette['text_color'])
    
    # Set up first subplot for accuracy
    format_axes_and_grid(axs[0], palette)
    axs[0].set_xticks(digit_weeks)
    axs[0].set_ylabel('Accuracy', color=palette['text_color'])
    axs[0].plot(
        digit_weeks, 
        val_accuracies_retrain_week_by_week, 
        label='Validation (retrain week by week)', 
        color=palette['first_color']
    )
    axs[0].plot(
        digit_weeks, 
        test_accuracies_retrain_week_by_week, 
        label=f"Test (retrain week by week) with final weeks: {weeks_test}", 
        color=palette['second_color']
    )

    axs[0].plot(
        digit_weeks,
        val_accuracies_test_on_trained_model_20_21,
        label="Validation (over model from 2020-2021)",
        color=palette['first_color'],
        linestyle='dotted'
    )

    axs[0].plot(
        digit_weeks,
        test_accuracies_test_on_trained_model_20_21,
        label=f"Test (over model from 2020-2021) with final weeks: {weeks_test}",
        color=palette['second_color'],
        linestyle='dotted'
    )

    axs[0].tick_params(axis='y')
    axs[0].legend(
        frameon=False, 
        labelcolor=palette['text_color'], 
        loc='upper left', 
        bbox_to_anchor=(1, 0.5)
    )

    # Set up second subplot for sample size
    format_axes_and_grid(axs[1], palette)
    axs[1].set_xticks(digit_weeks)
    axs[1].set_xlabel('Weeks', color=palette['text_color'])
    axs[1].set_ylabel('Sample Size', color=palette['text_color'])
    axs[1].bar(
        [week - 0.1 for week in digit_weeks], 
        val_total_retrain_week_by_week, 
        width=0.1,
        color=palette['first_color'], 
        label='Validation Set',
    )
    axs[1].bar(
        [week + 0.1 for week in digit_weeks], 
        test_total_retrain_week_by_week, 
        width=0.1, 
        color=palette['second_color'],  
        label='Test Set',
    )
    axs[1].tick_params(axis='y')
    axs[1].legend(
        frameon=False, 
        labelcolor=palette['text_color'], 
        loc='upper left', 
        bbox_to_anchor=(1, 0.5)
    )

    fig.tight_layout()
    fig.savefig(filename)
