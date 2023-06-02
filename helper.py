import matplotlib.pyplot as plt


def configure_matplotlib(labelsize: int = 18,
                         titlesize: int = 22,
                         titlepad: int = 25,
                         labelpad: int = 15,
                         tick_major_pad: int = 10,
                         dpi: int = 200,
                         platform: str = 'vscode') -> None:
    """
    Configures matplotlib to use the fivethirtyeight style and the Ubuntu font.
    Args:
        labelsize: The size of the axis labels
        titlesize: The size of the title
        titlepad: The padding of the title
        labelpad: The padding of the axis labels
        tick_major_pad: The padding of the major ticks
        dpi: The resolution of the figure
        platform: The platform on which the code is run (default: vscode)
    """
    plt.rcParams['font.family'] = 'Arial'
    plt.style.use('fivethirtyeight')
    plt.rcParams['font.serif'] = 'Ubuntu'
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = labelsize
    plt.rcParams['axes.labelpad'] = labelpad
    plt.rcParams['axes.titlesize'] = titlesize
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.titlepad'] = titlepad
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['xtick.major.pad'] = tick_major_pad
    plt.rcParams['ytick.major.pad'] = tick_major_pad

    # change the background color of the figure
    # plt.rcParams['figure.facecolor'] = 'none'
    # plt.rcParams['axes.facecolor'] = 'none'

    if platform == 'vscode':
        plt.rcParams.update({
            "figure.facecolor": (0.31, 0.31, 0.31, 0.39),
            "figure.edgecolor": (0.31, 0.31, 0.31, 0),
            "axes.facecolor": (0.31, 0.31, 0.31, 0),
            "axes.edgecolor": (0.31, 0.31, 0.31, 0.39),
            "text.color": "white",
            "axes.labelcolor": "white",
            "axes.titlecolor": "white",
        })
    elif platform == 'pycharm':
        plt.rcParams.update({
            "figure.facecolor": (0.31, 0.31, 0.31, 0),
            "figure.edgecolor": (0.31, 0.31, 0.31, 0.39),
            "axes.facecolor": (0.31, 0.31, 0.39, 0),
            "axes.edgecolor": (0.31, 0.31, 0.31, 0.39),
            "text.color": "white",
            "axes.labelcolor": "white",
            "axes.titlecolor": (0, 0, 0, 0.9),
        })

    # remove the top and right spines
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    # change the color of the grid
    plt.rcParams['axes.grid'] = False
