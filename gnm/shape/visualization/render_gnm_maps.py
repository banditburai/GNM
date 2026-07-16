# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Render GNM depth maps and camera-space normal maps.

`render_gnm.render_gnm` produces shaded RGB only. This module renders the
geometry buffers needed to condition generative image models (e.g.
ControlNet-style depth / normal conditioning): a metric depth map and a
camera-space normal map per camera, plus a foreground mask.

Conventions:
  * Cameras follow the same OpenCV-convention `world_to_camera` /
    `camera_to_image` matrices as `render_gnm` — the helpers
    `get_look_at_world_to_camera` / `get_fill_factor_camera_to_image` there
    can be reused directly.
  * Depth is the raw z-buffer depth in model units; 0 marks background.
  * Normals are unit camera-space normals (OpenCV camera axes: X right,
    Y down, Z towards the scene), encoded into RGB as `n * 0.5 + 0.5`.
"""

import dataclasses

import numpy as np

from gnm.shape import gnm_numpy
from gnm.shape.visualization import camera_conversions
from gnm.shape.visualization import gnm_pyrender
from gnm.shape.visualization import render_gnm

import pyrender  # pylint: disable=g-bad-import-order


@dataclasses.dataclass(frozen=True)
class GNMMaps:
  """Per-camera geometry buffers, batched over the leading dimension.

  Attributes:
    depth: Z-buffer depth in model units, 0 at background, (N, H, W).
    normal_rgb: Camera-space normals encoded as `n * 0.5 + 0.5`, float32
      [0-1], background pixels are 0.5 gray, (N, H, W, 3).
    mask: Foreground mask, bool, (N, H, W).
  """

  depth: np.ndarray
  normal_rgb: np.ndarray
  mask: np.ndarray

  def normalized_inverse_depth(self) -> np.ndarray:
    """Depth-Anything-style disparity map, per-frame normalized to [0, 1].

    Foreground disparity is scaled to [0, 1] (near = 1); background is 0.
    This is the common input format for depth-conditioned diffusion models.
    """
    disparity = np.zeros_like(self.depth)
    for i in range(self.depth.shape[0]):
      mask = self.mask[i]
      if not mask.any():
        continue
      inv = np.where(mask, 1.0 / np.maximum(self.depth[i], 1e-6), 0.0)
      lo, hi = inv[mask].min(), inv[mask].max()
      disparity[i] = np.where(mask, (inv - lo) / max(hi - lo, 1e-9), 0.0)
    return disparity


def render_gnm_maps(
    gnm_np: gnm_numpy.GNM,
    vertices: np.ndarray | None = None,
    world_to_camera: np.ndarray | None = None,
    camera_to_image: np.ndarray | None = None,
    image_size: tuple[int, int] = (512, 512),
    triangles: str | np.ndarray = '~eye_exteriors',
) -> GNMMaps:
  """Render depth and camera-space normal maps of GNM meshes.

  All array arguments broadcast over a single leading batch dimension N
  (frames/cameras), mirroring `render_gnm.render_gnm`.

  Args:
    gnm_np: The NumPy GNM model.
    vertices: Posed GNM vertices in world space, (V, 3) or (N, V, 3). Template
      vertices if not given.
    world_to_camera: OpenCV-convention world-to-camera transforms, (4, 4) or
      (N, 4, 4). Default look-at camera if not given.
    camera_to_image: OpenCV-convention intrinsics matrices, (4, 4) or
      (N, 4, 4). Default fill-factor intrinsics if not given.
    image_size: The width and height of the rendered maps in pixels: (W, H).
    triangles: A GNM vertex group name or an array of triangle indices,
      determining which triangles to render.

  Returns:
    A `GNMMaps` with (N, ...) leading batch dimension.
  """

  width, height = image_size

  if vertices is None:
    vertices = gnm_np.template_vertex_positions
  vertices = np.asarray(vertices, dtype=np.float32)
  if vertices.ndim == 2:
    vertices = vertices[None]

  if world_to_camera is None:
    world_to_camera = render_gnm.get_look_at_world_to_camera(
        gnm_np, vertices
    )
  if camera_to_image is None:
    camera_to_image = render_gnm.get_fill_factor_camera_to_image(
        gnm_np, vertices, image_size=image_size
    )

  world_to_camera = np.broadcast_to(
      np.asarray(world_to_camera, dtype=np.float32),
      (vertices.shape[0], 4, 4),
  )
  camera_to_image = np.broadcast_to(
      np.asarray(camera_to_image, dtype=np.float32),
      (vertices.shape[0], 4, 4),
  )
  num_frames = vertices.shape[0]

  triangle_indices = triangles
  if isinstance(triangles, str):
    triangle_indices = gnm_np.triangle_indices_for_group(triangles)
  faces = gnm_np.triangles[triangle_indices]

  # OpenGL-convention matrices for pyrender.
  world_to_camera_gl = camera_conversions.opencv_extrinsics_to_opengl(
      world_to_camera
  )
  camera_to_image_gl = (
      camera_conversions.opencv_intrinsics_matrix_to_opengl_view_matrix(
          camera_to_image,
          width=width,
          height=height,
          near=0.01,
          far=100.0,
      )
  )

  vertex_normals = gnm_np.compute_vertex_normals(vertices)

  scene = pyrender.Scene(
      bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[1.0, 1.0, 1.0]
  )
  camera = gnm_pyrender.ProjectionMatrixCamera(camera_to_image_gl[0].copy())
  camera_node = scene.add(camera, pose=np.linalg.inv(world_to_camera_gl[0]))

  renderer = pyrender.OffscreenRenderer(width, height)
  flags = pyrender.constants.RenderFlags.FLAT | (
      pyrender.constants.RenderFlags.SKIP_CULL_FACES
  )

  depths = np.zeros((num_frames, height, width), dtype=np.float32)
  normal_rgbs = np.full(
      (num_frames, height, width, 3), 0.5, dtype=np.float32
  )
  masks = np.zeros((num_frames, height, width), dtype=bool)

  mesh_node = None
  for f in range(num_frames):
    # Camera-space normals, encoded as colors. OpenCV convention: rotate world
    # normals by the extrinsics rotation.
    normals_cam = vertex_normals[f] @ world_to_camera[f, :3, :3].T
    normals_cam /= np.maximum(
        np.linalg.norm(normals_cam, axis=-1, keepdims=True), 1e-9
    )
    normal_colors = np.clip(normals_cam * 0.5 + 0.5, 0.0, 1.0).astype(
        np.float32
    )

    primitive = pyrender.Primitive(
        positions=vertices[f],
        indices=faces,
        normals=vertex_normals[f],
        color_0=normal_colors,
        material=pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, roughnessFactor=1.0
        ),
    )
    if mesh_node is not None:
      scene.remove_node(mesh_node)
    mesh_node = scene.add(pyrender.Mesh(primitives=[primitive]))

    scene.set_pose(camera_node, np.linalg.inv(world_to_camera_gl[f]))
    camera.set_projection_matrix(camera_to_image_gl[f])

    color, depth = renderer.render(scene, flags=flags)
    mask = depth > 0.0
    depths[f] = depth
    masks[f] = mask
    color = color.astype(np.float32) / 255.0
    normal_rgbs[f] = np.where(mask[..., None], color, 0.5)

  renderer.delete()
  return GNMMaps(depth=depths, normal_rgb=normal_rgbs, mask=masks)
