import numpy as np

from gen_ai.base.input import Input


class YOLOWorldInput(Input):
    """
    YOLOWorld input data.

    Parameters
    ----------
    image : np.ndarray
        Image to run detection on.
    """

    image: np.ndarray
