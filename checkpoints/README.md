# Fine-tuned checkpoints

The 80 checkpoints evaluated in the paper will be attached to a GitHub release. Put them here as

```
checkpoints/x4/<model>_<variant>.pth     srresnet rcan edsr realesrnet realesrgan swinir sinsr
checkpoints/x2/<model>_<variant>.pth     rcan edsr realesrgan swinir
```

and verify them with `sha256sum -c SHA256SUMS` (run in this folder). Variants: `wasp`,
`ctrl_{l1,lpips,dists,msssim}`, `abla_{la,lb,lc,lalb}` (x4, not SRResNet) and `x4/srresnet_l1.pth`
(the L1-trained SRResNet starting point). Each file is a plain PyTorch state dict for the model
returned by `models.build_model(<model>, <scale>)`; load it with `models.load_checkpoint`. SinSR
files hold the UNet only. See the main README for how each checkpoint was selected.
