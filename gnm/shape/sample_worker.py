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

"""CLI worker that samples GNM identity/expression params to an npz file.

This module loads TensorFlow (via the semantic samplers). TF must never share
a process with OSMesa rendering (their bundled LLVMs clash and segfault), so
run this as its own subprocess and feed the resulting npz to
`gnm.shape.visualization.render_worker`:

  python -m gnm.shape.sample_worker \
      --gender FEMALE --ethnicity ASIAN --expression HAPPY \
      --seed 0 --out /tmp/gnm_params.npz

Sampling is deterministic for a given --seed: the seed constructs a
`np.random.default_rng` Generator that is passed to the samplers. (Calling
`np.random.seed` would NOT work — the samplers default to a fresh
`np.random.default_rng()`, which the legacy global seed does not affect.)

Blends are also supported, e.g.:

  --expression-blend "HAPPY=0.7,SURPRISE=0.3"
  --gender-blend "FEMALE=0.8,MALE=0.2" --ethnicity-blend "ASIAN=1"
"""

import argparse
import enum
import json
from typing import TypeVar

import numpy as np

from gnm.shape import semantic_sampler

_E = TypeVar('_E', bound=enum.IntEnum)


def _parse_weights(spec: str, enum_cls: type[_E]) -> dict[_E, float]:
  """Parses 'NAME=W,NAME=W' into an enum-keyed weight mapping."""
  weights: dict[_E, float] = {}
  for item in spec.split(','):
    name, _, value = item.partition('=')
    name = name.strip().upper()
    if name not in enum_cls.__members__:
      raise ValueError(
          f'{name!r} is not a {enum_cls.__name__}; choose from '
          f'{list(enum_cls.__members__)}'
      )
    weights[enum_cls[name]] = float(value) if value else 1.0
  return weights


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  identity = parser.add_argument_group('identity')
  identity.add_argument(
      '--gender',
      default='FEMALE',
      choices=list(semantic_sampler.Gender.__members__),
  )
  identity.add_argument(
      '--ethnicity',
      default='WHITE',
      choices=list(semantic_sampler.Ethnicity.__members__),
  )
  identity.add_argument(
      '--gender-blend',
      default=None,
      help='e.g. "FEMALE=0.8,MALE=0.2"; overrides --gender.',
  )
  identity.add_argument(
      '--ethnicity-blend',
      default=None,
      help='e.g. "ASIAN=0.5,WHITE=0.5"; overrides --ethnicity.',
  )
  identity.add_argument('--num-identities', type=int, default=1)

  expression = parser.add_argument_group('expression')
  expression.add_argument(
      '--expression',
      default='HAPPY',
      choices=list(semantic_sampler.Expression.__members__),
  )
  expression.add_argument(
      '--expression-blend',
      default=None,
      help='e.g. "HAPPY=0.7,SURPRISE=0.3"; overrides --expression.',
  )
  expression.add_argument('--num-expressions', type=int, default=1)

  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--out', required=True, help='Output .npz path.')
  return parser


def sample_params(args: argparse.Namespace) -> dict[str, np.ndarray]:
  """Samples identity and expression arrays per the parsed CLI args."""
  rng = np.random.default_rng(args.seed)

  identity_sampler = semantic_sampler.IdentitySampler()
  if args.gender_blend or args.ethnicity_blend:
    gender_weights = _parse_weights(
        args.gender_blend or args.gender, semantic_sampler.Gender
    )
    ethnicity_weights = _parse_weights(
        args.ethnicity_blend or args.ethnicity, semantic_sampler.Ethnicity
    )
    identity = identity_sampler.blend_identities(
        gender_weights,
        ethnicity_weights,
        num_samples=args.num_identities,
        rng=rng,
    )
  else:
    identity = identity_sampler.sample_identity(
        semantic_sampler.Gender[args.gender],
        semantic_sampler.Ethnicity[args.ethnicity],
        num_samples=args.num_identities,
        rng=rng,
    )

  expression_sampler = semantic_sampler.ExpressionSampler()
  if args.expression_blend:
    expression = np.stack([
        expression_sampler.blend_expressions(
            _parse_weights(
                args.expression_blend, semantic_sampler.Expression
            ),
            rng=rng,
        )
        for _ in range(args.num_expressions)
    ])
  else:
    expression = expression_sampler.sample_expression(
        semantic_sampler.Expression[args.expression],
        num_samples=args.num_expressions,
        rng=rng,
    )

  meta = {
      'seed': args.seed,
      'gender': args.gender_blend or args.gender,
      'ethnicity': args.ethnicity_blend or args.ethnicity,
      'expression': args.expression_blend or args.expression,
  }
  return {
      'identity': np.asarray(identity, dtype=np.float32),
      'expression': np.asarray(expression, dtype=np.float32),
      'meta': np.array(json.dumps(meta)),
  }


def main(argv: list[str] | None = None) -> None:
  args = build_parser().parse_args(argv)
  arrays = sample_params(args)
  np.savez(args.out, **arrays)
  print(
      f'wrote {args.out}: identity {arrays["identity"].shape}, '
      f'expression {arrays["expression"].shape}'
  )


if __name__ == '__main__':
  main()
