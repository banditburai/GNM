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

"""Fit GNM identity to 2D facial landmarks from one or more photos.

Optimizes a SHARED identity plus per-photo expression and per-photo rigid
pose/camera against 68-point (iBUG-ordered) 2D landmarks, through the
differentiable pytorch backend. Multi-photo shared identity is the
regularizer that single-photo 3DMM fitting lacks.

Torch-only (no TensorFlow, no GL) — safe to run inside a marimo kernel.

Camera model per photo: OpenCV-style pinhole. The head stays at the origin;
each photo gets extrinsics (axis-angle `rvec`, translation `tvec`, camera
z forward) and a scalar focal length in pixels (principal point fixed at the
image center). Landmark coordinates are pixels, y down.

Typical use:

  import torch
  from gnm.shape import gnm_pytorch
  from gnm.shape.fitting_utils import fit_landmarks

  gnm_t = gnm_pytorch.GNM.from_local(...)
  result = fit_landmarks.fit_landmarks(gnm_t, landmarks_2d, image_sizes)
  vertices = gnm_t(torch.from_numpy(result.identity), ...)
"""

import dataclasses

import numpy as np
import torch

from gnm.shape import gnm_landmarks


@dataclasses.dataclass
class FitConfig:
  """Optimization settings.

  Attributes:
    pose_iters: Adam steps for stage A (rigid pose/camera only, mean face).
    full_iters: Adam steps for stage B (everything jointly).
    pose_lr: Learning rate for pose/camera parameters.
    shape_lr: Learning rate for identity/expression coefficients.
    identity_reg: L2 prior weight on identity coefficients.
    expression_reg: L2 prior weight on expression coefficients.
    optimize_focal: Whether to optimize the per-photo focal length.
    identity_components: If set, optimize only the first K identity
      coefficients (dominant shape modes) and hold the rest at zero. 68
      landmarks cannot constrain all 253 dims; restricting the subspace
      keeps the recovered signal in observable modes.
    device: Torch device string.
  """

  pose_iters: int = 400
  full_iters: int = 600
  pose_lr: float = 0.05
  shape_lr: float = 0.02
  identity_reg: float = 2e-4
  expression_reg: float = 1e-3
  optimize_focal: bool = True
  identity_components: int | None = None
  device: str = 'cpu'


@dataclasses.dataclass
class FitResult:
  """Fitted parameters and diagnostics.

  Attributes:
    identity: Shared identity coefficients, (identity_dim,).
    expression: Per-photo expression coefficients, (N, expression_dim).
    rvec: Per-photo axis-angle world-to-camera rotations, (N, 3).
    tvec: Per-photo world-to-camera translations, (N, 3).
    focal: Per-photo focal lengths in pixels, (N,).
    reprojection_rmse_px: Per-photo landmark RMSE in pixels, (N,).
    loss_history: Total loss per iteration over both stages.
  """

  identity: np.ndarray
  expression: np.ndarray
  rvec: np.ndarray
  tvec: np.ndarray
  focal: np.ndarray
  reprojection_rmse_px: np.ndarray
  loss_history: np.ndarray

  def world_to_camera(self) -> np.ndarray:
    """OpenCV-convention world-to-camera matrices, (N, 4, 4)."""
    rot = _rodrigues(torch.from_numpy(self.rvec)).numpy()
    out = np.tile(np.eye(4, dtype=np.float32), (self.rvec.shape[0], 1, 1))
    out[:, :3, :3] = rot
    out[:, :3, 3] = self.tvec
    return out

  def camera_to_image(self, image_sizes: np.ndarray) -> np.ndarray:
    """OpenCV-convention intrinsics matrices, (N, 4, 4)."""
    n = self.focal.shape[0]
    out = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    out[:, 0, 0] = self.focal
    out[:, 1, 1] = self.focal
    out[:, 0, 2] = image_sizes[:, 0] / 2.0
    out[:, 1, 2] = image_sizes[:, 1] / 2.0
    return out


def _rodrigues(rvec: torch.Tensor) -> torch.Tensor:
  """Axis-angle (..., 3) to rotation matrices (..., 3, 3)."""
  angle = torch.linalg.norm(rvec, dim=-1, keepdim=True).clamp_min(1e-9)
  axis = rvec / angle
  x, y, z = axis.unbind(-1)
  zero = torch.zeros_like(x)
  k = torch.stack(
      [zero, -z, y, z, zero, -x, -y, x, zero], dim=-1
  ).reshape(*rvec.shape[:-1], 3, 3)
  angle = angle[..., None]
  eye = torch.eye(3, dtype=rvec.dtype, device=rvec.device).expand_as(k)
  return eye + torch.sin(angle) * k + (1.0 - torch.cos(angle)) * (k @ k)


def _project(
    points: torch.Tensor,  # (N, L, 3) world
    rvec: torch.Tensor,  # (N, 3)
    tvec: torch.Tensor,  # (N, 3)
    focal: torch.Tensor,  # (N,)
    centers: torch.Tensor,  # (N, 2)
) -> torch.Tensor:
  """OpenCV pinhole projection to pixel coordinates, (N, L, 2)."""
  rot = _rodrigues(rvec)
  cam = torch.einsum('nij,nlj->nli', rot, points) + tvec[:, None, :]
  z = cam[..., 2:3].clamp_min(1e-6)
  return focal[:, None, None] * cam[..., :2] / z + centers[:, None, :]


def _interocular(landmarks_2d: torch.Tensor) -> torch.Tensor:
  """Distance between outer eye corners (iBUG 36, 45), (N,)."""
  return torch.linalg.norm(
      landmarks_2d[:, 45, :] - landmarks_2d[:, 36, :], dim=-1
  ).clamp_min(1.0)


def fit_landmarks(
    gnm_torch,
    landmarks_2d: np.ndarray,
    image_sizes: np.ndarray,
    config: FitConfig | None = None,
) -> FitResult:
  """Fits shared identity + per-photo expression/pose to 2D landmarks.

  Args:
    gnm_torch: A `gnm.shape.gnm_pytorch.GNM` (HEAD variant).
    landmarks_2d: Pixel-space landmarks, (N, 68, 2), iBUG order, y down.
    image_sizes: Per-photo (width, height), (N, 2).
    config: Optimization settings.

  Returns:
    A FitResult; `identity` is the person, reusable across poses/expressions.
  """
  config = config or FitConfig()
  device = torch.device(config.device)
  landmarks_2d = torch.as_tensor(
      landmarks_2d, dtype=torch.float32, device=device
  )
  image_sizes_np = np.asarray(image_sizes, dtype=np.float32)
  centers = torch.as_tensor(image_sizes_np / 2.0, device=device)
  num_photos = landmarks_2d.shape[0]

  # Head center from the template mean landmark: used to initialize
  # translation so the face starts in front of each camera.
  with torch.no_grad():
    _, template_lm = gnm_torch.vertices_and_landmarks(
        gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68
    )
    head_center = template_lm.mean(0)

  num_id = config.identity_components or gnm_torch.identity_dim
  num_id = min(num_id, gnm_torch.identity_dim)
  identity_free = torch.zeros(num_id, device=device, requires_grad=True)
  identity_tail = torch.zeros(
      gnm_torch.identity_dim - num_id, device=device
  )
  expression = torch.zeros(
      (num_photos, gnm_torch.expression_dim), device=device, requires_grad=True
  )
  rvec = torch.zeros((num_photos, 3), device=device)
  rvec += 1e-3  # Avoid the zero-angle singularity in Rodrigues.
  # OpenCV camera looks down +z with y down; the GNM head is y-up looking
  # +z. A 180-degree rotation about x maps head-up to image-up.
  rvec[:, 0] = np.pi
  rvec = rvec.clone().requires_grad_(True)
  tvec = torch.zeros((num_photos, 3), device=device)
  tvec[:, 2] = 2.0  # Model units are meters; ~2 m in front of the camera.
  tvec[:, :2] = -head_center[:2] * torch.tensor(
      [1.0, -1.0], device=device
  )  # Rotated head center to the optical axis.
  tvec = tvec.clone().requires_grad_(True)
  focal = torch.as_tensor(
      image_sizes_np[:, 0] * 1.5, device=device
  ).clone().requires_grad_(config.optimize_focal)

  interocular = _interocular(landmarks_2d)
  losses = []

  def reprojection_loss():
    identity = torch.cat([identity_free, identity_tail])
    _, lm3d = gnm_torch.vertices_and_landmarks(
        gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68,
        identity=identity[None, :].expand(num_photos, -1),
        expression=expression,
    )
    uv = _project(lm3d, rvec, tvec, focal, centers)
    err = torch.linalg.norm(uv - landmarks_2d, dim=-1)  # (N, 68) px
    normalized = (err / interocular[:, None]) ** 2
    return normalized.mean(), err

  # Stage A: rigid pose + camera only, mean face.
  pose_params = [rvec, tvec] + ([focal] if config.optimize_focal else [])
  opt = torch.optim.Adam(pose_params, lr=config.pose_lr)
  for _ in range(config.pose_iters):
    opt.zero_grad()
    loss, _ = reprojection_loss()
    loss.backward()
    opt.step()
    losses.append(float(loss))

  # Stage B: everything jointly.
  opt = torch.optim.Adam(
      [
          {'params': pose_params, 'lr': config.pose_lr * 0.2},
          {'params': [identity_free, expression], 'lr': config.shape_lr},
      ]
  )
  for _ in range(config.full_iters):
    opt.zero_grad()
    data_loss, _ = reprojection_loss()
    loss = (
        data_loss
        + config.identity_reg * identity_free.square().mean()
        + config.expression_reg * expression.square().mean()
    )
    loss.backward()
    opt.step()
    losses.append(float(loss))

  with torch.no_grad():
    _, err_px = reprojection_loss()
    rmse = err_px.square().mean(dim=-1).sqrt()

  return FitResult(
      identity=torch.cat([identity_free, identity_tail]).detach().cpu().numpy(),
      expression=expression.detach().cpu().numpy(),
      rvec=rvec.detach().cpu().numpy(),
      tvec=tvec.detach().cpu().numpy(),
      focal=focal.detach().cpu().numpy(),
      reprojection_rmse_px=rmse.cpu().numpy(),
      loss_history=np.asarray(losses, dtype=np.float32),
  )
