from typing import List, Optional, Tuple

from pydantic import BaseModel

from gen_ai.logger import logger

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


class Keypoint(BaseModel):
    """
    Represents a keypoint in a pose.

    Parameters
    ----------
    x : float
        The x-coordinate of the keypoint.
    y : float
        The y-coordinate of the keypoint.
    """

    x: float
    y: float


class Pose(BaseModel):
    """
    Represents a pose.

    Parameters
    ----------
    keypoints : List[Keypoint]
        The keypoints of the pose.
    """

    keypoints: List[Keypoint]

    def _get_keypoint(self, keypoint_name: str) -> Optional[Tuple[float, float]]:
        for idx, kp_name in KEYPOINT_MAPPING.items():
            if kp_name == keypoint_name:
                point = self.keypoints[idx]
                return (point.x, point.y) if (point.x, point.y) != (0, 0) else None
        return None

    def __getattr__(self, name: str) -> Optional[Tuple[float, float]]:
        if name in self.__annotations__ or (
            name.startswith("__") and name.endswith("__")
        ):
            return super().__getattr__(name)

        keypoint_name = name.replace("_", " ").title()
        result = self._get_keypoint(keypoint_name)
        if result is None:
            logger.error(f"'Pose' object has no attribute '{name}'. Returning 'None'.")
        return result


class Poses(BaseModel):
    """
    Represents a list of poses.

    Parameters
    ----------
    poses : List[Pose]
        The list of poses.
    """

    poses: List[Pose]

    def __getitem__(self, idx: int) -> Pose:
        return self.poses[idx]

    def __len__(self) -> int:
        return len(self.poses)
