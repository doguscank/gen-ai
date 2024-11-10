from ultralytics.engine.results import Results

from gen_ai.pose.pose import Keypoint, Pose, Poses


def parse_yolov11_pose_output(results: Results) -> Poses:
    """
    Parse the keypoints from the given results.

    Parameters
    ----------
    results : Results
        Results object containing the keypoints.

    Returns
    -------
    Poses
        List of Pose objects representing detected objects.
    """

    poses = results[0].keypoints.xy.cpu().numpy().astype(int)
    poses_obj = Poses(
        poses=[Pose(keypoints=[Keypoint(x=point[0], y=point[1]) for point in poses])]
    )

    return poses_obj
