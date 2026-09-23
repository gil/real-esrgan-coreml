"""Convert PyTorch Real-ESRGAN models to CoreML .mlpackage format.

Requires: torch, coremltools (one-time conversion, not needed at runtime).
Usage: uv run python convert.py --model x4plus --size 522
"""

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

WEIGHTS_DIR = Path(__file__).parent / "weights"

MODEL_CONFIGS = {
    "x4plus": dict(
        arch="rrdb", num_in_ch=3, num_out_ch=3, scale=4,
        num_feat=64, num_block=23, num_grow_ch=32,
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        pth_stem="RealESRGAN_x4plus",
    ),
    "x2plus": dict(
        arch="rrdb", num_in_ch=3, num_out_ch=3, scale=2,
        num_feat=64, num_block=23, num_grow_ch=32,
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        pth_stem="RealESRGAN_x2plus",
    ),
    "anime_6B": dict(
        arch="rrdb", num_in_ch=3, num_out_ch=3, scale=4,
        num_feat=64, num_block=6, num_grow_ch=32,
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        pth_stem="RealESRGAN_x4plus_anime_6B",
    ),
    "animevideo": dict(
        arch="srvgg", num_in_ch=3, num_out_ch=3,
        num_feat=64, num_conv=16, upscale=4,
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        pth_stem="realesr-animevideov3",
    ),
    "general": dict(
        arch="srvgg", num_in_ch=3, num_out_ch=3,
        num_feat=64, num_conv=32, upscale=4,
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        pth_stem="realesr-general-x4v3",
    ),
    "nomos2_otf": dict(
        arch="rrdb", num_in_ch=3, num_out_ch=3, scale=4,
        num_feat=64, num_block=23, num_grow_ch=32,
        keymap="esrgan_old",
        pth_stem="4xNomos2_otf_esrgan",
    ),
    "vhs_2x": dict(
        arch="rrdb", num_in_ch=3, num_out_ch=3, scale=2,
        num_feat=64, num_block=23, num_grow_ch=32,
        unshuffle=False, num_upsample=1,
        keymap="esrgan_old",
        pth_stem="2x_VHS-upscale-and-denoise_Film_477000_G",
    ),
    "genesis_cleanup": dict(
        arch="srvgg", num_in_ch=3, num_out_ch=3,
        num_feat=96, num_conv=24, upscale=1,
        pth_stem="Genesis_Cleanup_compact",
    ),
}


def download_pth(model_name: str) -> Path:
    """Download PyTorch .pth weights if not cached."""
    config = MODEL_CONFIGS[model_name]
    pth_path = WEIGHTS_DIR / f"{config['pth_stem']}.pth"
    if pth_path.exists():
        print(f"Found cached {pth_path}")
        return pth_path
    url = config.get("url")
    if not url:
        raise FileNotFoundError(
            f"{pth_path} not found and {model_name} has no download url. "
            f"Place the .pth there manually."
        )
    WEIGHTS_DIR.mkdir(exist_ok=True)
    print(f"Downloading {url} ...")

    def progress(count, block_size, total_size):
        pct = count * block_size * 100 // total_size
        print(f"\r  {pct}%", end="", flush=True)

    urlretrieve(url, pth_path, reporthook=progress)
    print()
    return pth_path


def build_torch_rrdb(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23,
                     num_grow_ch=32, unshuffle=None, num_upsample=2):
    """Build PyTorch RRDBNet without importing basicsr (inline definition).

    Real-ESRGAN's x2plus reaches 2x by pixel_unshuffling the input then running
    two upsample blocks. Original-ESRGAN 2x weights instead feed 3 channels
    straight in and use a single upsample block, so both knobs are explicit.
    """
    import torch
    import torch.nn as tnn
    import torch.nn.functional as F

    class _RDB(tnn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = tnn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
            self.conv2 = tnn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv3 = tnn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv4 = tnn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv5 = tnn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
            self.lrelu = tnn.LeakyReLU(0.2, True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class _RRDB(tnn.Module):
        def __init__(self):
            super().__init__()
            self.rdb1 = _RDB()
            self.rdb2 = _RDB()
            self.rdb3 = _RDB()

        def forward(self, x):
            out = self.rdb1(x)
            out = self.rdb2(out)
            out = self.rdb3(out)
            return out * 0.2 + x

    do_unshuffle = (scale == 2) if unshuffle is None else unshuffle

    class _RRDBNet(tnn.Module):
        def __init__(self):
            super().__init__()
            self.unshuffle = do_unshuffle
            self.num_upsample = num_upsample
            first_in_ch = num_in_ch * 4 if do_unshuffle else num_in_ch
            self.conv_first = tnn.Conv2d(first_in_ch, num_feat, 3, 1, 1)
            self.body = tnn.Sequential(*[_RRDB() for _ in range(num_block)])
            self.conv_body = tnn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1 = tnn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if num_upsample >= 2:
                self.conv_up2 = tnn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = tnn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = tnn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = tnn.LeakyReLU(0.2, True)

        def forward(self, x):
            if self.unshuffle:
                x = F.pixel_unshuffle(x, 2)
            feat = self.conv_first(x)
            body_feat = self.conv_body(self.body(feat))
            feat = feat + body_feat
            feat = self.lrelu(self.conv_up1(
                torch.nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
            if self.num_upsample >= 2:
                feat = self.lrelu(self.conv_up2(
                    torch.nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    return _RRDBNet()


def build_torch_srvgg(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4):
    """Build PyTorch SRVGGNetCompact without external dependencies."""
    import torch
    import torch.nn as tnn
    import torch.nn.functional as F

    class _SRVGGNetCompact(tnn.Module):
        def __init__(self):
            super().__init__()
            self.num_in_ch = num_in_ch
            self.num_out_ch = num_out_ch
            self.num_feat = num_feat
            self.num_conv = num_conv
            self.upscale = upscale

            self.body = tnn.ModuleList()
            self.body.append(tnn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
            self.body.append(tnn.PReLU(num_parameters=num_feat))
            for _ in range(num_conv):
                self.body.append(tnn.Conv2d(num_feat, num_feat, 3, 1, 1))
                self.body.append(tnn.PReLU(num_parameters=num_feat))
            self.body.append(tnn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
            self.upsampler = tnn.PixelShuffle(upscale)

        def forward(self, x):
            out = x
            for i in range(len(self.body)):
                out = self.body[i](out)
            # At 1x the pixel shuffle and the nearest resize are both identities;
            # skipping them keeps the traced graph free of no-op resize layers.
            if self.upscale > 1:
                out = self.upsampler(out)
                out = out + F.interpolate(x, scale_factor=self.upscale, mode='nearest')
            else:
                out = out + x
            return out

    return _SRVGGNetCompact()


def remap_esrgan_old(state_dict, num_upsample=2):
    """Rename old-ESRGAN keys to RRDBNet attribute names.

    Checkpoints from the original ESRGAN repo (and most OpenModelDB weights)
    name layers by nn.Sequential index: model.0, model.1.sub.N.RDBj.convk.0,
    then the tail. The last sub index is conv_body, not an RRDB block. Each
    upsample block shifts the tail indices, so 2x and 4x nets differ there.
    """
    import re

    if num_upsample >= 2:
        tail = {"3": "conv_up1", "6": "conv_up2", "8": "conv_hr", "10": "conv_last"}
    else:
        tail = {"3": "conv_up1", "5": "conv_hr", "7": "conv_last"}
    sub_ids = [int(m.group(1)) for k in state_dict
               for m in [re.match(r"model\.1\.sub\.(\d+)\.", k)] if m]
    trunk = max(sub_ids)

    out = {}
    for k, v in state_dict.items():
        if k.startswith("model.0."):
            nk = "conv_first." + k[len("model.0."):]
        elif k.startswith(f"model.1.sub.{trunk}."):
            nk = "conv_body." + k[len(f"model.1.sub.{trunk}."):]
        elif k.startswith("model.1.sub."):
            nk, n = re.subn(
                r"model\.1\.sub\.(\d+)\.RDB(\d)\.conv(\d)\.0\.",
                lambda m: f"body.{m.group(1)}.rdb{m.group(2)}.conv{m.group(3)}.", k)
            if n == 0:
                raise ValueError(f"unmapped ESRGAN key: {k}")
        else:
            m = re.match(r"model\.(\d+)\.", k)
            if not m or m.group(1) not in tail:
                raise ValueError(f"unmapped ESRGAN key: {k}")
            nk = tail[m.group(1)] + "." + k[m.end():]
        out[nk] = v
    return out


def get_mlpackage_name(model_name: str, input_size: int, fp16: bool) -> str:
    """Return .mlpackage filename for a model."""
    dtype_label = "fp16" if fp16 else "fp32"
    return f"RealESRGAN_{model_name}_{input_size}_{dtype_label}.mlpackage"


def convert(model_name: str = "x4plus", input_size: int = 512, use_fp16: bool = True):
    """Convert a Real-ESRGAN model to CoreML .mlpackage."""
    import torch
    import coremltools as ct

    config = MODEL_CONFIGS[model_name]
    arch = config["arch"]
    scale = config.get("scale", config.get("upscale", 4))

    pth_path = download_pth(model_name)

    # Build PyTorch model
    print(f"Building PyTorch model ({model_name}, arch={arch})...")
    if arch == "rrdb":
        model = build_torch_rrdb(
            num_in_ch=config["num_in_ch"], num_out_ch=config["num_out_ch"],
            scale=config["scale"], num_feat=config["num_feat"],
            num_block=config["num_block"], num_grow_ch=config["num_grow_ch"],
            unshuffle=config.get("unshuffle"), num_upsample=config.get("num_upsample", 2),
        )
    else:
        model = build_torch_srvgg(
            num_in_ch=config["num_in_ch"], num_out_ch=config["num_out_ch"],
            num_feat=config["num_feat"], num_conv=config["num_conv"],
            upscale=config["upscale"],
        )

    checkpoint = torch.load(str(pth_path), map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
    if config.get("keymap") == "esrgan_old":
        state_dict = remap_esrgan_old(state_dict, config.get("num_upsample", 2))
    # Community weights are often stored fp16; the traced module is fp32.
    state_dict = {k: (v.float() if v.is_floating_point() else v)
                  for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Trace
    print(f"Tracing with input shape [1, 3, {input_size}, {input_size}]...")
    example_input = torch.randn(1, 3, input_size, input_size)
    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)

    # Convert to CoreML
    precision = ct.precision.FLOAT16 if use_fp16 else ct.precision.FLOAT32
    dtype_label = "fp16" if use_fp16 else "fp32"
    print(f"Converting to CoreML ({dtype_label})...")

    ct_model = ct.convert(
        traced,
        inputs=[ct.TensorType(shape=(1, 3, input_size, input_size), name="input")],
        compute_precision=precision,
        minimum_deployment_target=ct.target.macOS15,
    )

    # Save
    WEIGHTS_DIR.mkdir(exist_ok=True)
    out_name = get_mlpackage_name(model_name, input_size, use_fp16)
    out_path = WEIGHTS_DIR / out_name
    ct_model.save(str(out_path))
    print(f"Saved: {out_path}")

    # Verify output shape
    print("Verifying output shape...")
    test_input = np.random.randn(1, 3, input_size, input_size).astype(np.float32)
    result = ct_model.predict({"input": test_input})
    out_key = list(result.keys())[0]
    out_shape = result[out_key].shape
    expected = (1, 3, input_size * scale, input_size * scale)
    assert out_shape == expected, f"Shape mismatch: {out_shape} != {expected}"
    print(f"Output shape OK: {out_shape}")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Convert Real-ESRGAN to CoreML")
    parser.add_argument("--model", default="x4plus", choices=list(MODEL_CONFIGS.keys()),
                        help="Model variant (default: x4plus)")
    parser.add_argument("--size", type=int, default=512,
                        help="Input spatial size for fixed-shape model (default: 512)")
    parser.add_argument("--fp32", action="store_true", help="Use float32 instead of float16")
    args = parser.parse_args()
    convert(model_name=args.model, input_size=args.size, use_fp16=not args.fp32)


if __name__ == "__main__":
    main()
