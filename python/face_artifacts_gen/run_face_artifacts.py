#!/usr/bin/env python3
"""
Generate natural face artifacts (beard, glasses with glare, bangs, occluders)
on a folder of images.

Examples:

  # CPU procedural baseline (no GPU / no heavy deps)
  python run_face_artifacts.py -i /data/faces -o /data/out \\
      --method procedural --artifact glasses_glare --limit 20

  # FLUX.1 Fill inpainting (needs ~24GB GPU + HF access)
  python run_face_artifacts.py -i /data/faces -o /data/out \\
      --method flux_fill --artifact beard --device cuda --cpu-offload

  # SDXL + ControlNet
  python run_face_artifacts.py -i /data/faces -o /data/out \\
      --method sdxl_cn --artifact bangs --device cuda

  # Instruction edit (Qwen / FLUX Kontext) with paste-back
  python run_face_artifacts.py -i /data/faces -o /data/out \\
      --method instruct_edit --artifact glasses_glare --device cuda

  # StyleGAN latent edit (or aligned-crop fallback)
  python run_face_artifacts.py -i /data/faces -o /data/out \\
      --method stylegan --artifact beard --device cuda
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from fa.constants import ARTIFACTS, METHODS
from fa.pipeline import process_folder


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Generate natural face artifacts on an image folder.',
    )
    p.add_argument('-i', '--input-root', type=Path, required=True, help='Input dataset root')
    p.add_argument('-o', '--output-root', type=Path, required=True, help='Output root')
    p.add_argument(
        '--method',
        choices=METHODS,
        default='procedural',
        help='Generation backend',
    )
    p.add_argument(
        '--artifact',
        choices=ARTIFACTS,
        required=True,
        help='Artifact type to generate',
    )
    p.add_argument('--device', default='cuda', help='cuda | mps | cpu')
    p.add_argument('--limit', type=int, default=0, help='Process at most N images (0=all)')
    p.add_argument('--seed', type=int, default=None, help='Base RNG seed')
    p.add_argument(
        '--occluded-policy',
        choices=('keep', 'missing'),
        default=None,
        help='How to mark keypoints under the artifact mask',
    )
    p.add_argument('--max-nme', type=float, default=0.08, help='QA landmark drift threshold')
    p.add_argument('--skip-qa', action='store_true', help='Skip landmark NME filter')
    p.add_argument('--use-clip', action='store_true', help='Compute CLIP artifact score')
    p.add_argument('--no-save-masks', action='store_true')
    p.add_argument('--require-markup', action='store_true')
    p.add_argument('-v', '--verbose', action='store_true')

    # Method-specific knobs (forwarded into generator_kwargs).
    p.add_argument('--model-id', default=None, help='Override HF model id')
    p.add_argument('--steps', type=int, default=None)
    p.add_argument('--guidance', type=float, default=None)
    p.add_argument('--strength', type=float, default=None)
    p.add_argument('--inference-size', type=int, default=None)
    p.add_argument('--dtype', default=None, help='float16 | bfloat16 | float32')
    p.add_argument('--cpu-offload', action='store_true')
    p.add_argument(
        '--instruct-backend',
        choices=('qwen', 'flux_kontext', 'procedural_fallback'),
        default=None,
    )
    p.add_argument('--control-mode', choices=('canny', 'depth'), default='canny')
    p.add_argument('--assets-root', type=Path, default=None)
    return p


def generator_kwargs_from_args(args: argparse.Namespace) -> dict:
    kw = {}
    if args.model_id:
        kw['model_id'] = args.model_id
    if args.steps is not None:
        kw['steps'] = args.steps
    if args.guidance is not None:
        kw['guidance'] = args.guidance
    if args.strength is not None:
        kw['strength'] = args.strength
    if args.inference_size is not None:
        kw['inference_size'] = args.inference_size
    if args.dtype:
        kw['dtype'] = args.dtype
    if args.cpu_offload:
        kw['cpu_offload'] = True
    if args.instruct_backend:
        kw['backend'] = args.instruct_backend
    if args.control_mode:
        kw['control_mode'] = args.control_mode
    if args.assets_root:
        kw['assets_root'] = str(args.assets_root)
    return kw


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    if not args.input_root.exists():
        logging.error('input-root does not exist: %s', args.input_root)
        return 2

    counts = process_folder(
        input_root=args.input_root,
        output_root=args.output_root,
        method=args.method,
        artifact=args.artifact,
        device=args.device,
        limit=args.limit,
        seed=args.seed,
        occluded_policy=args.occluded_policy,
        max_nme=args.max_nme,
        skip_qa=args.skip_qa,
        use_clip=args.use_clip,
        save_masks=not args.no_save_masks,
        require_markup=args.require_markup,
        generator_kwargs=generator_kwargs_from_args(args),
    )
    logging.info('summary: %s', counts)
    return 0 if counts.get('ok', 0) > 0 or counts.get('total', 0) == 0 else 0


if __name__ == '__main__':
    sys.exit(main())
