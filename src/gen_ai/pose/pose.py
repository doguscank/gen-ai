from typing import List, Tuple, Optional


KEYPOINT_MAPPING = {
    0: "Nose",
    1: "Left Eye",
    2: "Right Eye",
    3: "Left Ear",
    4: "Right Ear",
    5: "Left Shoulder",
    6: "Right Shoulder",
    7: "Left Elbow",
    8: "Right Elbow",
    9: "Left Wrist",
    10: "Right Wrist",
    11: "Left Hip",
    12: "Right Hip",
    13: "Left Knee",
    14: "Right Knee",
    15: "Left Ankle",
    16: "Right Ankle",
}


class Pose:
    def __init__(self, keypoints: List[Tuple[float, float]]):
        """
        Initialize the Pose with the given keypoints.

        Parameters
        ----------
        keypoints : List[Tuple[float, float]]
            List of keypoints where each keypoint is a tuple (x, y).
        """
        self.keypoints = keypoints

    def __getattr__(self, name: str) -> Optional[Tuple[float, float]]:
        """
        Get the coordinates of the keypoint by its name.

        Parameters
        ----------
        name : str
            Name of the keypoint attribute.

        Returns
        -------
        Optional[Tuple[float, float]]
            Tuple (x, y) coordinates of the keypoint or None if not found.

        Raises
        ------
        AttributeError
            If the keypoint name is not found.
        """
        keypoint_name = name.replace("_", " ").title()
        for idx, kp_name in KEYPOINT_MAPPING.items():
            if kp_name == keypoint_name:
                x, y = self.keypoints[idx]
                return (x, y) if (x, y) != (0, 0) else None
        raise AttributeError(f"'Pose' object has no attribute '{name}'")
