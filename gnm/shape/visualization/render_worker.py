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

"""CLI worker that renders a GNM params npz into images, maps, or a GLB.

This module creates GL contexts (OSMesa on headless Linux). It must never
share a process with TensorFlow (their bundled LLVMs clash and segfault) and
cannot run on a marimo kernel thread — run it as its own subprocess, fed by
`gnm.shape.sample_worker` (or any npz with `identity` and `expression`):

  python -m gnm.shape.visualization.render_worker \
      --params /tmp/gnm_params.npz --views 5 --max-azimuth 60 --res 256 \
      --sheet-out /tmp/gnm_sheet.npy

Outputs (any combination; at least one is required):
  --sheet-out   (H, W, 3) uint8 contact sheet, rows = shaded RGB / normalized
                inverse depth / camera-space normals, one column per view.
  --maps-out    npz with rgb, depth, disparity, normal_rgb, mask, plus the
                world_to_camera / camera_to_image matrices and azimuth angles
                (everything needed to reproject 3D points into each view).
  --glb-out     posed mesh exported via trimesh (geometry only, no texture).

The params npz may also carry `joint_rotations` (num_joints, 3) and
`translation` (3,) to pose the head; they default to zeros.
"""

import argparse
import json

import numpy as np

from gnm.shape import gnm_numpy


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--params',
      required=True,
      help='npz with identity/expression (from sample_worker or a fit).',
  )
  parser.add_argument(
      '--identity-index',
      type=int,
      default=0,
      help='Row to use when the npz holds multiple identities.',
  )
  parser.add_argument(
      '--expression-index',
      type=int,
      default=0,
      help='Row to use when the npz holds multiple expressions.',
  )

  camera = parser.add_argument_group('camera')
  camera.add_argument('--views', type=int, default=5)
  camera.add_argument(
      '--max-azimuth',
      type=float,
      default=60.0,
      help='Views sweep azimuth over [-max, +max] degrees.',
  )
  camera.add_argument(
      '--polar', type=float, default=0.0, help='Polar angle in degrees.'
  )
  camera.add_argument(
      '--camera-distance',
      type=float,
      default=None,
      help='Camera distance in model units (model default if omitted).',
  )
  camera.add_argument(
      '--fill-factor',
      type=float,
      default=None,
      help='Target head fill factor in the frame (model default if omitted).',
  )
  camera.add_argument('--res', type=int, default=256)

  outputs = parser.add_argument_group('outputs')
  outputs.add_argument('--sheet-out', default=None)
  outputs.add_argument('--maps-out', default=None)
  outputs.add_argument('--glb-out', default=None)
  return parser


def _load_params(args: argparse.Namespace, gnm: gnm_numpy.GNM) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
  """Loads (identity, expression, joint_rotations, translation) from the npz."""
  data = np.load(args.params)
  identity = np.atleast_2d(data['identity'])[args.identity_index]
  expression = np.atleast_2d(data['expression'])[args.expression_index]
  joint_rotations = (
      data['joint_rotations']
      if 'joint_rotations' in data
      else np.zeros((gnm.num_joints, 3))
  )
  translation = data['translation'] if 'translation' in data else np.zeros(3)
  return identity, expression, joint_rotations, translation


def main(argv: list[str] | None = None) -> None:
  args = build_parser().parse_args(argv)
  if not (args.sheet_out or args.maps_out or args.glb_out):
    raise SystemExit('specify at least one of --sheet-out/--maps-out/--glb-out')

  # Import GL-touching modules lazily so --help works anywhere; gnm_pyrender
  # selects OSMesa on headless Linux (respecting a preset PYOPENGL_PLATFORM).
  from gnm.shape.visualization import render_gnm
  from gnm.shape.visualization import render_gnm_maps

  gnm = gnm_numpy.GNM.from_local(
      version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD
  )
  identity, expression, joint_rotations, translation = _load_params(args, gnm)
  vertices = gnm(identity, expression, joint_rotations, translation).astype(
      np.float32
  )

  if args.glb_out:
    import trimesh

    mesh = trimesh.Trimesh(
        vertices=vertices, faces=gnm.triangles, process=False
    )
    mesh.export(args.glb_out)
    print(f'wrote {args.glb_out}')

  if not (args.sheet_out or args.maps_out):
    return

  num_views = args.views
  size = (args.res, args.res)
  azimuths = np.linspace(-args.max_azimuth, args.max_azimuth, num_views)[
      :, None
  ]
  batched = np.broadcast_to(vertices, (num_views, *vertices.shape))

  look_at_kwargs = dict(azimuthal_angle=azimuths, polar_angle=args.polar)
  fill_kwargs = dict(image_size=size)
  if args.camera_distance is not None:
    look_at_kwargs['camera_distance'] = args.camera_distance
    fill_kwargs['camera_distance'] = args.camera_distance
  if args.fill_factor is not None:
    fill_kwargs['target_fill_factor'] = args.fill_factor

  world_to_camera = render_gnm.get_look_at_world_to_camera(
      gnm, batched, **look_at_kwargs
  )
  camera_to_image = render_gnm.get_fill_factor_camera_to_image(
      gnm, batched, **fill_kwargs
  )

  rgb = render_gnm.render_gnm(
      gnm, batched, world_to_camera, camera_to_image, image_size=size
  )
  maps = render_gnm_maps.render_gnm_maps(
      gnm, batched, world_to_camera, camera_to_image, image_size=size
  )
  disparity = maps.normalized_inverse_depth()

  if args.maps_out:
    np.savez_compressed(
        args.maps_out,
        rgb=np.asarray(rgb, dtype=np.float32),
        depth=maps.depth,
        disparity=disparity.astype(np.float32),
        normal_rgb=maps.normal_rgb,
        mask=maps.mask,
        world_to_camera=world_to_camera.astype(np.float32),
        camera_to_image=camera_to_image.astype(np.float32),
        azimuth=azimuths.astype(np.float32),
        meta=np.array(json.dumps(vars(args), default=str)),
    )
    print(f'wrote {args.maps_out}')

  if args.sheet_out:
    rows = [
        np.concatenate([np.clip(x, 0, 1) ** (1 / 2.2) for x in rgb], axis=1),
        np.concatenate(
            [np.repeat(d[..., None], 3, axis=-1) for d in disparity], axis=1
        ),
        np.concatenate(list(maps.normal_rgb), axis=1),
    ]
    sheet = (np.concatenate(rows, axis=0) * 255).astype(np.uint8)
    np.save(args.sheet_out, sheet)
    print(f'wrote {args.sheet_out}: {sheet.shape}')


if __name__ == '__main__':
  main()
