"""
# Plotting Utilities

This module provides utilities for setting up plotting environment.



"""

import dataclasses
import matplotlib as mpl
import matplotlib.pyplot as plt


@dataclasses.dataclass
class PlotPallette:
    """Plotting color pallette

    Attributes:
        bg_rgba_color (str): Background color in rgba format
        bg_color (str): Background color
        fg_color (str): Foreground color
        primary_color (str): Primary color
        secondary_color (str): Secondary color
        tertiary_color (str): Tertiary color
        quaternary_color (str): Quaternary color
        plotly_template (str): Plotly template
        matplot_template (str): Matplotlib template
    """

    bg_rgba_color: str = "rgba(38,42,50,1.0)"
    bg_color: str = "#262a32"
    fg_color: str = "#ffffff"
    primary_color: str = "#11acd5"
    secondary_color: str = "#ce6cff"
    tertiary_color: str = "#ea3424"
    quaternary_color: str = "#5cc99a"
    plotly_template: str = "plotly_dark"
    matplot_template: str = "dark_background"

    @property
    def colors(self):
        """Get color pallette"""
        return [self.primary_color, self.secondary_color, self.tertiary_color, self.quaternary_color]


light_theme = PlotPallette(
    bg_rgba_color="rgba(255,255,255,1.0)",
    bg_color="#ffffff",
    fg_color="#000000",
    primary_color="#11acd5",
    secondary_color="#ce6cff",
    tertiary_color="#ea3424",
    quaternary_color="#5cc99a",
    plotly_template="plotly",
    matplot_template="default",
)

dark_theme = PlotPallette(
    bg_rgba_color="rgba(38,42,50,1.0)",
    bg_color="#262a32",
    fg_color="#ffffff",
    primary_color="#11acd5",
    secondary_color="#ce6cff",
    tertiary_color="#ea3424",
    quaternary_color="#5cc99a",
    plotly_template="plotly_dark",
    matplot_template="dark_background",
)


def setup_plotting(theme: PlotPallette = dark_theme) -> PlotPallette:
    """Setup plotting environment for matplotlib and plotly

    Args:
        theme (PlotPallette, optional): Plotting theme. Defaults to dark_theme.

    Returns:
        PlotPallette: Plotting theme

    Example:

    ```python
    import heartkit as hk

    plot_theme = hk.util.ssetup_plotting(hk.utils.light_theme)
    """
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.style.use(theme.matplot_template)
    mpl.rcParams["axes.facecolor"] = theme.bg_color
    mpl.rcParams["figure.facecolor"] = theme.bg_color
    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
    return theme
