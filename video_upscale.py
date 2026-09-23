"""Optimized video upscaling pipeline with fully pipelined I/O and inference.

Threaded read (PNG decode + tile prep) -> main thread inference -> threaded save (uint8 + PNG encode).
All I/O overlaps with GPU inference.

Several models can be chained with --model a,b,c. Frames stay as PNG between
passes, so a chain costs no extra encode generations.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np
from PIL import Image

from upscale import (
    MODEL_CONFIGS,
    PRE_PAD,
    _blend_weight,
    _prepare_tile,
    _unpad_tile,
    compute_tile_starts,
    load_coreml_model,
)

FRAME_GLOB = "frame_*.png"
FRAME_PATTERN = "frame_%06d.png"


def resolve_chain(model_names: list[str]):
    """Plan a multi-model run. Returns ([(name, scale, pre_downscale)], final_scale).

    Output scale is capped at the largest single model scale rather than the
    product, so two 2x models still land at 2x. The trim happens *before* a pass
    so peak resolution never exceeds the cap either.
    """
    scales = [MODEL_CONFIGS[m]["scale"] for m in model_names]
    cap = max(scales)
    plan = []
    cur = 1
    for name, scale in zip(model_names, scales):
        factor = 1
        if cur * scale > cap:
            target = cap // scale
            if target < 1 or cur % target != 0:
                raise ValueError(
                    f"cannot fit {name} ({scale}x) after a {cur}x result "
                    f"within a {cap}x cap without a fractional resize"
                )
            factor = cur // target
            cur = target
        plan.append((name, scale, factor))
        cur *= scale
    return plan, cur


def _box_downscale(rgb: np.ndarray, factor: int) -> np.ndarray:
    """Area-average downscale. Exact in float when the size divides evenly."""
    h, w = rgb.shape[:2]
    if h % factor == 0 and w % factor == 0:
        return rgb.reshape(h // factor, factor, w // factor, factor, 3).mean(axis=(1, 3))
    img = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), "RGB")
    img = img.resize((w // factor, h // factor), Image.BOX)
    return np.array(img, dtype=np.float32) / 255.0


def _save_frame_png(output_path, frame_out):
    """Save a frame as PNG with fast compression."""
    out_uint8 = np.clip(frame_out * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(out_uint8, "RGB").save(output_path, compress_level=1)


def process_frames_with_io(
    frames_dir: Path,
    output_dir: Path,
    model_name: str,
    compute_unit: str,
    fp16: bool,
    model_size: int,
    scale: int,
    tile_size: int,
    tile_overlap: int,
    model=None,
    out_key: str = None,
    downscale: int = 1,
    return_outputs: bool = True,
    label: str = "Pipelined",
) -> list[np.ndarray]:
    """Pipelined: threaded read -> inference -> threaded save.

    Returns list of output float32 arrays for quality comparison. Pass
    return_outputs=False to keep them out of memory; at 2x a 30s chunk of
    output frames is several GB.
    """
    if model is None or out_key is None:
        model, out_key = load_coreml_model(model_name, model_size, compute_unit, fp16)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(frames_dir.glob(FRAME_GLOB))
    n = len(paths)

    # Get frame dimensions from first file
    first_img = Image.open(paths[0]).convert("RGB")
    h, w = first_img.size[1], first_img.size[0]
    h, w = h // downscale, w // downscale
    c = 3
    out_h, out_w = h * scale, w * scale
    single_tile = (h <= tile_size and w <= tile_size)

    # Precompute tile layout
    y_starts = compute_tile_starts(h, tile_size, tile_overlap)
    x_starts = compute_tile_starts(w, tile_size, tile_overlap)
    tile_specs = []
    for y0 in y_starts:
        for x0 in x_starts:
            tile_specs.append((y0, x0, min(y0 + tile_size, h), min(x0 + tile_size, w)))
    tiles_per_frame = len(tile_specs)

    # Precompute blend weights
    blend_weights = []
    if not single_tile:
        for y0, x0, y1, x1 in tile_specs:
            th, tw = y1 - y0, x1 - x0
            blend_weights.append(_blend_weight(th, tw, y0, x0, y1, x1, h, w, scale, tile_overlap))

    # --- Read thread: decode PNGs and prepare tile inputs ---
    read_queue = Queue(maxsize=4)

    def read_worker():
        for fi, p in enumerate(paths):
            img = Image.open(p).convert("RGB")
            rgb = np.array(img, dtype=np.float32) / 255.0
            if downscale > 1:
                rgb = _box_downscale(rgb, downscale)
            tile_inputs = []
            if single_tile:
                chw, ph, pw, th, tw = _prepare_tile(rgb, 0, 0, h, w, model_size, PRE_PAD)
                tile_inputs.append((chw[None], ph, pw, th, tw, 0))
            else:
                for ti, (y0, x0, y1, x1) in enumerate(tile_specs):
                    chw, ph, pw, th, tw = _prepare_tile(rgb, y0, x0, y1, x1, model_size, PRE_PAD)
                    tile_inputs.append((chw[None], ph, pw, th, tw, ti))
            read_queue.put((fi, p.name, tile_inputs))
        read_queue.put(None)

    # --- Save pool ---
    save_pool = ThreadPoolExecutor(max_workers=8)
    save_futures = []

    read_thread = Thread(target=read_worker, daemon=True)
    read_thread.start()

    outputs = [None] * n if return_outputs else None
    # Pre-allocate input buffer (reused for every predict call)
    input_buf = np.empty((1, c, model_size, model_size), dtype=np.float32)

    while True:
        item = read_queue.get()
        if item is None:
            break

        fi, filename, tile_inputs = item

        if single_tile:
            x, ph, pw, th, tw, ti = tile_inputs[0]
            input_buf[0] = x[0]
            result = model.predict({"input": input_buf})
            out_nchw = result[out_key][0]
            frame_out = _unpad_tile(out_nchw, ph, pw, scale, PRE_PAD).astype(np.float32)
        else:
            frame_out = np.zeros((out_h, out_w, c), dtype=np.float32)
            weight_map = np.zeros((out_h, out_w, 1), dtype=np.float32)
            for x, ph, pw, th, tw, ti in tile_inputs:
                input_buf[0] = x[0]
                result = model.predict({"input": input_buf})
                out_nchw = result[out_key][0]
                tile_out = _unpad_tile(out_nchw, ph, pw, scale, PRE_PAD)
                y0, x0 = tile_specs[ti][0], tile_specs[ti][1]
                bw = blend_weights[ti]
                oy0, ox0 = y0 * scale, x0 * scale
                frame_out[oy0:oy0 + th * scale, ox0:ox0 + tw * scale, :] += tile_out * bw
                weight_map[oy0:oy0 + th * scale, ox0:ox0 + tw * scale, :] += bw
            frame_out = np.clip(frame_out / np.maximum(weight_map, 1e-8), 0.0, 1.0).astype(np.float32)

        if return_outputs:
            outputs[fi] = frame_out
        save_futures.append(save_pool.submit(_save_frame_png, str(output_dir / filename), frame_out))
        print(f"\r{label}: frame {fi+1}/{n}", end="", flush=True)

    # Wait for saves
    for f in save_futures:
        f.result()
    save_pool.shutdown(wait=True)
    read_thread.join()
    print()
    return outputs


def run_chain(src_dir: Path, work: Path, plan, fp16: bool, tile_size: int,
              tile_overlap: int, keep_passes: bool, fit: bool = False) -> Path:
    """Run every pass over PNG frames. Returns the directory holding the result.

    With fit=True each pass gets a tile as large as that pass's own input, so a
    frame is one inference with no seams. Passes can need different sizes, so
    the size is resolved per pass rather than once for the whole chain.
    """
    cur = src_dir
    total = len(plan)
    if fit:
        fw, fh = Image.open(sorted(src_dir.glob(FRAME_GLOB))[0]).size
    for i, (name, scale, factor) in enumerate(plan, 1):
        out_dir = work / f"pass{i}_{name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        label = f"Pass {i}/{total} [{name}]"
        if factor > 1:
            print(f"{label}: area downscale by {factor} before this pass")
        if fit:
            fw, fh = fw // factor, fh // factor
            pass_tile = max(fw, fh)
            print(f"{label}: fit, tile {pass_tile} for a {fw}x{fh} frame")
        else:
            pass_tile = tile_size
        t0 = time.time()
        process_frames_with_io(
            cur, out_dir, name, "CPU_AND_GPU", fp16,
            pass_tile + PRE_PAD * 2, scale, pass_tile, tile_overlap,
            downscale=factor, return_outputs=False, label=label,
        )
        if fit:
            fw, fh = fw * scale, fh * scale
        print(f"{label}: {time.time() - t0:.1f}s")
        if not keep_passes:
            shutil.rmtree(cur, ignore_errors=True)
        cur = out_dir

    final_dir = work / "final"
    if final_dir.exists():
        shutil.rmtree(final_dir, ignore_errors=True)
    cur.rename(final_dir)
    return final_dir


def main():
    parser = argparse.ArgumentParser(description="Optimized video upscaling with pipelined I/O")
    parser.add_argument("input", help="Input video/GIF path")
    parser.add_argument("-o", "--output", help="Output video path (.mp4)")
    parser.add_argument("--frames-dir", help="Write PNG frames here instead of encoding a video")
    parser.add_argument("--model", default="x4plus",
                        help="Model name, or a comma separated chain (e.g. genesis_cleanup,vhs_2x)")
    parser.add_argument("--keep-passes", action="store_true",
                        help="Keep each pass's frames instead of deleting as it goes")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=32)
    parser.add_argument("--fit", action="store_true",
                        help="Size the model to each pass's frame: one inference, no seams. "
                             "Ignores --tile-size. Converts a model per size on first use.")
    args = parser.parse_args()

    if not args.output and not args.frames_dir:
        parser.error("need -o/--output or --frames-dir")

    models = [m.strip() for m in args.model.split(",") if m.strip()]
    if not models:
        parser.error("--model needs at least one name")
    unknown = [m for m in models if m not in MODEL_CONFIGS]
    if unknown:
        parser.error(f"unknown model(s): {', '.join(unknown)}. "
                     f"available: {', '.join(MODEL_CONFIGS)}")

    try:
        plan, final_scale = resolve_chain(models)
    except ValueError as e:
        parser.error(str(e))

    input_path = Path(args.input)

    # Get video info
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(input_path)],
        capture_output=True, text=True
    )
    streams = json.loads(probe.stdout)["streams"]
    video_stream = next(s for s in streams if s["codec_type"] == "video")
    fps = video_stream["r_frame_rate"]
    w, h = int(video_stream["width"]), int(video_stream["height"])

    chain_desc = " -> ".join(f"{n}({s}x)" for n, s, _ in plan)
    print(f"Chain: {chain_desc}, final {final_scale}x")

    owned_tmp = None
    if args.frames_dir:
        work = Path(args.frames_dir)
        work.mkdir(parents=True, exist_ok=True)
    else:
        owned_tmp = Path(tempfile.mkdtemp(prefix="esrgan_"))
        work = owned_tmp

    try:
        src_dir = work / "src"
        src_dir.mkdir(parents=True, exist_ok=True)

        print(f"Extracting frames from {input_path} ({w}x{h}, {fps} fps)...")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(input_path), str(src_dir / FRAME_PATTERN)],
            capture_output=True
        )
        n_frames = len(list(src_dir.glob(FRAME_GLOB)))
        print(f"Extracted {n_frames} frames")

        print("Upscaling (fit)..." if args.fit else f"Upscaling (tile={args.tile_size})...")
        t0 = time.time()
        final_dir = run_chain(src_dir, work, plan, True,
                              args.tile_size, args.tile_overlap, args.keep_passes,
                              fit=args.fit)
        elapsed = time.time() - t0
        print(f"Upscaled {n_frames} frames in {elapsed:.1f}s ({elapsed/max(n_frames,1):.2f}s/frame)")

        if args.output:
            print("Encoding output video...")
            subprocess.run([
                "ffmpeg", "-y", "-framerate", fps,
                "-i", str(final_dir / FRAME_PATTERN),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "fast",
                str(args.output),
            ], capture_output=True)

        out_w, out_h = w * final_scale, h * final_scale
        dest = args.output if args.output else final_dir
        print(f"Done: {w}x{h} -> {out_w}x{out_h}, saved to {dest}")
    finally:
        if owned_tmp is not None:
            shutil.rmtree(owned_tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
