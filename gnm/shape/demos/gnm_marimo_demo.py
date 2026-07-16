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

"""Marimo port of the GNM head demos.

Run with:  marimo edit gnm/shape/demos/gnm_marimo_demo.py

Unlike the Jupyter demos (ipywidgets + THREE.js browser viewer), this notebook
renders server-side with pyrender, so it also works headless (OSMesa/EGL) on
remote GPU boxes. In addition to the shaded RGB view it renders the
depth / normal-map conditioning buffers from
`gnm.shape.visualization.render_gnm_maps` for each camera of a multi-angle
sweep — the output format intended for conditioning generative image models.
"""

import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell
def _():
    from gnm.shape import gnm_numpy
    from gnm.shape import semantic_sampler
    from gnm.shape.visualization import render_gnm
    from gnm.shape.visualization import render_gnm_maps

    gnm = gnm_numpy.GNM.from_local(
        version=gnm_numpy.GNMMajorVersion.V3,
        variant=gnm_numpy.GNMVariant.HEAD,
    )
    return gnm, render_gnm, render_gnm_maps, semantic_sampler


@app.cell
def _(mo, semantic_sampler):
    # Semantic sampling controls (mirrors semantic_gnm_demo.ipynb).
    expression_label = mo.ui.dropdown(
        options=[e.name for e in semantic_sampler.Expression],
        value="HAPPY",
        label="Expression",
    )
    gender = mo.ui.dropdown(
        options=[g.name for g in semantic_sampler.Gender],
        value="FEMALE",
        label="Gender",
    )
    ethnicity = mo.ui.dropdown(
        options=[e.name for e in semantic_sampler.Ethnicity],
        value="ASIAN",
        label="Ethnicity",
    )
    seed = mo.ui.number(value=0, label="Seed")
    use_template = mo.ui.checkbox(value=False, label="Template (zero params)")
    mo.hstack([expression_label, gender, ethnicity, seed, use_template])
    return ethnicity, expression_label, gender, seed, use_template


@app.cell
def _(mo, np):
    # Pose + camera sweep controls (mirrors gnm_head_demo.ipynb pose sliders).
    neck_yaw = mo.ui.slider(-40, 40, value=0, step=5, label="Neck yaw (deg)")
    head_pitch = mo.ui.slider(-30, 30, value=0, step=5, label="Head pitch (deg)")
    gaze_yaw = mo.ui.slider(-25, 25, value=0, step=5, label="Gaze yaw (deg)")
    num_views = mo.ui.slider(1, 9, value=5, step=2, label="Camera views")
    max_azimuth = mo.ui.slider(10, 90, value=60, step=10, label="Max azimuth (deg)")
    resolution = mo.ui.dropdown(
        options={"256": 256, "384": 384, "512": 512}, value="256", label="Res"
    )
    mo.hstack(
        [neck_yaw, head_pitch, gaze_yaw, num_views, max_azimuth, resolution]
    )
    _ = np
    return gaze_yaw, head_pitch, max_azimuth, neck_yaw, num_views, resolution


@app.cell
def _(
    ethnicity,
    expression_label,
    gender,
    np,
    seed,
    semantic_sampler,
    use_template,
):
    # Sample identity + expression parameters.
    _rng_seed = int(seed.value)
    if use_template.value:
        identity_params = None
        expression_params = None
    else:
        np.random.seed(_rng_seed)
        _id_sampler = semantic_sampler.IdentitySampler()
        _expr_sampler = semantic_sampler.ExpressionSampler()
        identity_params = _id_sampler.sample_identity(
            semantic_sampler.Gender[gender.value],
            semantic_sampler.Ethnicity[ethnicity.value],
            num_samples=1,
        )[0]
        expression_params = _expr_sampler.sample_expression(
            semantic_sampler.Expression[expression_label.value],
            num_samples=1,
        )[0]
    return expression_params, identity_params


@app.cell
def _(
    expression_params,
    gaze_yaw,
    gnm,
    head_pitch,
    identity_params,
    neck_yaw,
    np,
):
    # Pose the mesh.
    rotations = np.zeros((gnm.num_joints, 3), dtype=np.float32)
    # Joints: 0=neck, 1=head, 2=left eye, 3=right eye (axis-angle, radians).
    rotations[0, 1] = np.deg2rad(neck_yaw.value)
    rotations[1, 0] = np.deg2rad(head_pitch.value)
    rotations[2, 1] = rotations[3, 1] = np.deg2rad(gaze_yaw.value)

    vertices = gnm(
        identity_params, expression_params, rotations, np.zeros(3)
    ).astype(np.float32)
    return (vertices,)


@app.cell
def _(
    gnm,
    max_azimuth,
    mo,
    np,
    num_views,
    render_gnm,
    render_gnm_maps,
    resolution,
    vertices,
):
    # Multi-angle sweep: shaded RGB + depth + normal conditioning maps.
    _n = int(num_views.value)
    _res = int(resolution.value)
    _az = np.linspace(-max_azimuth.value, max_azimuth.value, _n)[:, None]
    _vb = np.broadcast_to(vertices, (_n, *vertices.shape))

    _w2c = render_gnm.get_look_at_world_to_camera(
        gnm, _vb, azimuthal_angle=_az
    )
    _c2i = render_gnm.get_fill_factor_camera_to_image(
        gnm, _vb, image_size=(_res, _res)
    )

    _rgb = render_gnm.render_gnm(
        gnm, _vb, _w2c, _c2i, image_size=(_res, _res)
    )
    _maps = render_gnm_maps.render_gnm_maps(
        gnm, _vb, _w2c, _c2i, image_size=(_res, _res)
    )
    _disp = _maps.normalized_inverse_depth()

    _row_rgb = np.concatenate(
        [np.clip(_r, 0, 1) ** (1 / 2.2) for _r in _rgb], axis=1
    )
    _row_dep = np.concatenate(
        [np.repeat(_d[..., None], 3, axis=-1) for _d in _disp], axis=1
    )
    _row_nrm = np.concatenate(list(_maps.normal_rgb), axis=1)
    sheet = (
        np.concatenate([_row_rgb, _row_dep, _row_nrm], axis=0) * 255
    ).astype(np.uint8)

    mo.vstack(
        [
            mo.md("**Rows:** shaded RGB / normalized inverse depth /"
                  " camera-space normals"),
            mo.image(sheet),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
