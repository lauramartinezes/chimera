import matplotlib

from .config import *


def format_axes_and_grid(ax, palette):
    ax.spines['left'].set_color(palette['grid_color'])
    ax.spines['bottom'].set_color(palette['grid_color'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(bottom=False, left=False, labelleft=True, labelbottom=True)
    ax.tick_params(axis='both', colors=palette['text_color'])
    ax.grid(visible=True, axis='y', color=palette['grid_color'])
    ax.grid(visible=True, axis='x', color=palette['grid_color'])


def select_charts_font(font_name=FONT_KULEUVEN, font_size=None):
    # Check all available fonts in:
    # https://jonathansoma.com/lede/data-studio/matplotlib/list-all-fonts-available-in-matplotlib-plus-samples/

    try:
        font_family_name = "sans-serif"
        matplotlib.rcParams['font.family'] = font_family_name
        matplotlib.rcParams['font.' + font_family_name] = font_name
    except:
        font_family_name = "serif"
        matplotlib.rcParams['font.family'] = font_family_name
        matplotlib.rcParams['font.' + font_family_name] = font_name

    if font_size is not None:
        matplotlib.rcParams.update({'font.size': font_size})
