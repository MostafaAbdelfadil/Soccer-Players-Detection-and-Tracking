import cv2
import numpy as np

class PerspectiveTransformation:
    """ This a class for transforming image between front view and top view

    Attributes:
        src (np.array): Coordinates of 4 source points
        dst (np.array): Coordinates of 4 destination points
        M (np.array): Matrix to transform image from front view to top view
        M_inv (np.array): Matrix to transform image from top view to front view
    """
    def __init__(self):
        """Init PerspectiveTransformation."""
        
        self.src = np.float32([(1197, 80),     # top-left
                            (62, 505),     # bottom-left
                            (4141, 541),    # bottom-right
                            (3007, 85)])    # top-right
        self.dst = np.float32([(95, 85),
                            (95, 911),
                            (1405, 911),
                            (1405, 85)])
        
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)