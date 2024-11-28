from typing import Optional, Tuple

import numpy as np

from gen_ai.logger import logger


def show_model(self, mesh_color: Optional[Tuple[float, float, float]] = None) -> None:
    """
    Show the model in 3D using open3d

    Parameters
    ----------
    mesh_color: Tuple[float, float, float], optional
        Color of the mesh. Default is None

    Returns
    -------
    None
    """

    try:
        import open3d as o3d  # pylint: disable=import-outside-toplevel
    except ImportError:
        logger.error("Please install open3d with 'pip install open3d'")

    with open("result.obj", "w", encoding="utf-8") as f:
        f.write(self.obj_data)

    mesh = o3d.io.read_triangle_mesh("result.obj")
    mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices)
    if mesh_color is not None and len(mesh_color) == 3:
        colors = np.ones((len(vertices), 3)) * mesh_color
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([mesh])  # pylint: disable=no-member
