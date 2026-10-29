# Third-party code and models

`pose_synth` itself contains only glue code. The methods it calls are cloned into
`third_party/` by `envs/install_*.sh` and are not part of this repository.
Their code and, separately, their weights come with their own terms. Several
restrict use to **non-commercial research**, so synthesized images and any
model trained on them may inherit those limits. Check this before using results
outside of experiments.

| Method | Repository (pinned commit) | Code licence | Weights and assets |
|---|---|---|---|
| 3DDFA-V2 | [cleardusk/3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) `1b6c676` | MIT | Built on the Basel Face Model (BFM), whose licence is non-commercial research only |
| LivePortrait | [KlingAIResearch/LivePortrait](https://github.com/KlingAIResearch/LivePortrait) `9b294b3` | MIT | The InsightFace detector and landmark models it downloads are non-commercial research only |
| GPAvatar | [xg-chu/GPAvatar](https://github.com/xg-chu/GPAvatar) `3ca7d44` | MIT | FLAME (MPI licence, non-commercial), EMOCA and MICA assets (non-commercial), plus the InsightFace models used by MICA |
| FaceLift | [weijielyu/FaceLift](https://github.com/weijielyu/FaceLift) `0a9420c` | Apache-2.0 and Adobe Research License v1.2 | The Adobe Research License covers non-commercial research only, including redistribution of derivatives |
| DiffPortrait360 | [FreedomGu/Diffportrait360](https://github.com/FreedomGu/Diffportrait360) `91106e7` | Apache-2.0 | PanoHead weights (CC BY-NC-SA 4.0, non-commercial) and the Stable Diffusion 1.5 base (CreativeML Open RAIL-M) |

Local modifications applied by the install scripts:

- 3DDFA-V2 has `np.long` replaced with `np.int64` in `bfm/bfm.py` and `np.int`
  replaced with `np.intp` in `FaceBoxes/utils/nms/cpu_nms.pyx`, so it works with
  current numpy.
- GPAvatar gets a symlink `inference_tools/checkpoints/one_model.ckpt` that
  points to `checkpoints/one_model.ckpt`.
