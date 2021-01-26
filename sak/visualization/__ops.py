import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def get_fig_kwargs(**kwargs):
    # Retrieve axis
    if ("axis" in kwargs) or ("axes" in kwargs) or ("ax" in kwargs):
        if "axis" in kwargs:
            ax = kwargs["axis"]
        elif "axes" in kwargs:
            ax = kwargs["axes"]
        elif "ax" in kwargs:
            ax = kwargs["ax"]

        # Solve for ax = None
        if ax is None:
            # Retrieve figure
            if "figure" in kwargs:
                f = kwargs["figure"]
            elif "fig" in kwargs:
                f = kwargs["fig"]
            elif "f" in kwargs:
                f = kwargs["f"]
            else:
                f = plt.figure(**kwargs)

            # Solve for figure = None
            if f is None:
                f = plt.figure(**kwargs)

            ax = f.gca()
        else:
            f = None
    else:
        # Retrieve figure
        if "figure" in kwargs:
            f = kwargs["figure"]
        elif "fig" in kwargs:
            f = kwargs["fig"]
        elif "f" in kwargs:
            f = kwargs["f"]
        else:
            f = plt.figure(**kwargs)

        # Solve for figure = None
        if f is None:
            f = plt.figure(**kwargs)

        ax = f.gca()

    return f,ax