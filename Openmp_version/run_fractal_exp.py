import os
import sys
import time
import json
import logging
import argparse
import platform
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="HPC Parallel Fractal Zoom Movie (Version 0 & Version 1)")
    parser.add_argument("--experiments_name", "-experiments_name", type=str, required=True)
    parser.add_argument("--results_folder", "-results_folder", type=str, default="results")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--bench_frames", type=int, default=10)
    parser.add_argument("--tile_size", type=int, default=20)
    parser.add_argument("--p_list", "-p_list", type=str, default="1,2,4,8,16,32,64,128,256")
    parser.add_argument("--run_repetition", "-run_repetition", type=int, default=3)
    parser.add_argument("--zoom_center_x", type=float, default=-0.3436)
    parser.add_argument("--zoom_center_y", type=float, default=0.5318)
    parser.add_argument("--zoom_factor", type=float, default=0.97)
    parser.add_argument("--min_iter", type=int, default=200)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--threading_layer", type=str, default="workqueue", choices=["omp", "workqueue", "tbb", "default"])
    return parser.parse_args()

args = parse_args()
p_values = [int(p.strip()) for p in args.p_list.split(",") if p.strip()]

exp_dir = os.path.join(args.results_folder, args.experiments_name)
frames_dir = os.path.join(exp_dir, "frames")
os.makedirs(exp_dir, exist_ok=True)
os.makedirs(frames_dir, exist_ok=True)

logical_cores = os.cpu_count() or 1
os.environ["NUMBA_NUM_THREADS"] = str(max(max(p_values), logical_cores))
if args.threading_layer != "default":
    os.environ["NUMBA_THREADING_LAYER"] = args.threading_layer

import numpy as np
import pandas as pd
import numba
from numba import njit, prange
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

def setup_logger():
    logger = logging.getLogger("Fractal_Benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(exp_dir, "experiment.log"), mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

logger = setup_logger()

@njit(parallel=False, cache=True)
def mandelbrot_serial(x0, x1, y0, y1, w, h, maxit, out):
    for j in range(h):
        y = y0 + (y1 - y0) * j / h
        for i in range(w):
            x = x0 + (x1 - x0) * i / w
            zr = 0.0
            zi = 0.0
            k = 0
            while k < maxit and (zr * zr + zi * zi) <= 4.0:
                zr_next = zr * zr - zi * zi + x
                zi = 2.0 * zr * zi + y
                zr = zr_next
                k += 1
            out[j, i] = k

@njit(parallel=True, cache=True)
def mandelbrot_variant_a(x0, x1, y0, y1, w, h, maxit, out):
    for j in prange(h):
        y = y0 + (y1 - y0) * j / h
        for i in range(w):
            x = x0 + (x1 - x0) * i / w
            zr = 0.0
            zi = 0.0
            k = 0
            while k < maxit and (zr * zr + zi * zi) <= 4.0:
                zr_next = zr * zr - zi * zi + x
                zi = 2.0 * zr * zi + y
                zr = zr_next
                k += 1
            out[j, i] = k

@njit(parallel=True, cache=True)
def mandelbrot_variant_b(x0, x1, y0, y1, w, h, maxit, tile_coords, out):
    num_tiles = tile_coords.shape[0]
    for t in prange(num_tiles):
        j_start = tile_coords[t, 0]
        j_end = tile_coords[t, 1]
        i_start = tile_coords[t, 2]
        i_end = tile_coords[t, 3]

        for j in range(j_start, j_end):
            y = y0 + (y1 - y0) * j / h
            for i in range(i_start, i_end):
                x = x0 + (x1 - x0) * i / w
                zr = 0.0
                zi = 0.0
                k = 0
                while k < maxit and (zr * zr + zi * zi) <= 4.0:
                    zr_next = zr * zr - zi * zi + x
                    zi = 2.0 * zr * zi + y
                    zr = zr_next
                    k += 1
                out[j, i] = k

def build_shuffled_tiles(w, h, tile_size):
    tiles = []
    for j in range(0, h, tile_size):
        j_end = min(j + tile_size, h)
        for i in range(0, w, tile_size):
            i_end = min(i + tile_size, w)
            tiles.append((j, j_end, i, i_end))

    tiles_arr = np.array(tiles, dtype=np.int32)
    np.random.seed(42)
    np.random.shuffle(tiles_arr)
    return tiles_arr

def get_frame_bounds(frame_idx, total_frames, w, h):
    base_x_span = 3.0
    base_y_span = base_x_span * (h / w)
    current_scale = args.zoom_factor ** frame_idx

    x_span = base_x_span * current_scale
    y_span = base_y_span * current_scale

    x0 = args.zoom_center_x - x_span / 2.0
    x1 = args.zoom_center_x + x_span / 2.0
    y0 = args.zoom_center_y - y_span / 2.0
    y1 = args.zoom_center_y + y_span / 2.0

    ratio = frame_idx / max(1, total_frames - 1)
    maxit = int(args.min_iter + (args.max_iter - args.min_iter) * ratio)
    return x0, x1, y0, y1, maxit

def warmup():
    logger.info("Performing Numba JIT compiler warm-up...")
    dummy = np.zeros((16, 16), dtype=np.int32)
    tiles = build_shuffled_tiles(16, 16, 4)
    mandelbrot_serial(-2.0, 1.0, -1.0, 1.0, 16, 16, 100, dummy)
    try:
        numba.set_num_threads(1)
        mandelbrot_variant_a(-2.0, 1.0, -1.0, 1.0, 16, 16, 100, dummy)
        mandelbrot_variant_b(-2.0, 1.0, -1.0, 1.0, 16, 16, 100, tiles, dummy)
        active_layer = numba.threading_layer()
        logger.info(f"Numba warm-up complete. Active threading layer: {active_layer}")
    except ValueError:
        os.environ["NUMBA_THREADING_LAYER"] = "default"
        numba.config.THREADING_LAYER = "default"
        mandelbrot_variant_a(-2.0, 1.0, -1.0, 1.0, 16, 16, 100, dummy)
        mandelbrot_variant_b(-2.0, 1.0, -1.0, 1.0, 16, 16, 100, tiles, dummy)
        active_layer = numba.threading_layer()
        logger.info(f"Warm-up finished with default layer: {active_layer}")
    return active_layer

def save_config(active_layer):
    config = {
        "experiment_name": args.experiments_name,
        "timestamp": datetime.now().isoformat(),
        "dimensions": {"width": args.width, "height": args.height},
        "frames": {"total_movie_frames": args.frames, "benchmark_frames": args.bench_frames},
        "tile_size": args.tile_size,
        "zoom": {
            "center": [args.zoom_center_x, args.zoom_center_y],
            "factor": args.zoom_factor,
            "min_iter": args.min_iter,
            "max_iter": args.max_iter
        },
        "p_list": p_values,
        "run_repetition": args.run_repetition,
        "hardware": {
            "logical_cores": os.cpu_count(),
            "threading_layer": active_layer,
            "os": platform.system(),
            "python": platform.python_version()
        }
    }
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    logger.info("Configuration saved to config.json")

def generate_zoom_movie():
    logger.info(f"Generating full {args.frames}-frame Zoom Movie (isolated from benchmarks)...")
    t0 = time.perf_counter()
    w, h = args.width, args.height
    img = np.zeros((h, w), dtype=np.int32)
    saved_frames = []

    for f_idx in range(args.frames):
        x0, x1, y0, y1, maxit = get_frame_bounds(f_idx, args.frames, w, h)
        mandelbrot_serial(x0, x1, y0, y1, w, h, maxit, img)

        fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=100)
        ax.imshow(img, cmap="twilight_shifted", origin="lower", aspect="auto")
        ax.axis("off")
        ax.text(0.02, 0.92, f"Zoom x{1.0 / (args.zoom_factor ** f_idx):.1f} | Frame {f_idx + 1}/{args.frames}",
                transform=ax.transAxes, color="white", fontsize=8, weight="bold",
                bbox=dict(facecolor="black", alpha=0.5, pad=2, edgecolor="none"))
        fig.tight_layout(pad=0)
        frame_file = os.path.join(frames_dir, f"frame_{f_idx:04d}.png")
        fig.savefig(frame_file)
        plt.close(fig)
        saved_frames.append(frame_file)

    gif_path = os.path.join(exp_dir, "fractal_zoom.gif")
    images = [imageio.imread(f) for f in saved_frames]
    imageio.mimsave(gif_path, images, fps=24, loop=0)
    logger.info(f"Zoom movie assembled at: {gif_path} (Time: {time.perf_counter() - t0:.2f}s)")

def generate_tile_cost_heatmap():
    w, h, ts = args.width, args.height, args.tile_size
    ny = (h + ts - 1) // ts
    nx = (w + ts - 1) // ts
    cost_grid = np.zeros((ny, nx), dtype=np.float64)

    x0, x1, y0, y1, maxit = get_frame_bounds(args.frames // 2, args.frames, w, h)
    frame = np.zeros((h, w), dtype=np.int32)
    mandelbrot_serial(x0, x1, y0, y1, w, h, maxit, frame)

    for j_idx, j in enumerate(range(0, h, ts)):
        j_end = min(j + ts, h)
        for i_idx, i in enumerate(range(0, w, ts)):
            i_end = min(i + ts, w)
            cost_grid[j_idx, i_idx] = np.mean(frame[j:j_end, i:i_end])

    plt.figure(figsize=(7, 4.5))
    plt.imshow(cost_grid, cmap="inferno", origin="lower")
    plt.colorbar(label="Mean Iterations per Tile")
    plt.title(f"Heatmap of Computation Cost per Tile (Frame {args.frames // 2})")
    plt.xlabel("Tile Column Index")
    plt.ylabel("Tile Row Index")
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "cost_per_tile_heatmap.png"), dpi=300)
    plt.close()
    logger.info("Saved cost_per_tile_heatmap.png")

def generate_plots(df):
    plt.figure(figsize=(8, 6))
    for v, grp in df[df["p"] > 1].groupby("version"):
        plt.plot(grp["p"], grp["speedup"], marker="o", linewidth=2, label=v)
    plt.axvline(x=logical_cores, color="red", linestyle=":", alpha=0.7, label=f"Logical Cores ({logical_cores})")
    plt.title(f"Mandelbrot Speedup vs Threads ({args.experiments_name})")
    plt.xlabel("Number of Threads (p)")
    plt.ylabel("Speedup (S)")
    plt.xticks(p_values)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "speedup.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    for v, grp in df[df["p"] > 1].groupby("version"):
        plt.plot(grp["p"], grp["efficiency"], marker="s", linewidth=2, label=v)
    plt.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="Ideal Efficiency")
    plt.axvline(x=logical_cores, color="red", linestyle=":", alpha=0.7, label=f"Logical Cores ({logical_cores})")
    plt.title(f"Mandelbrot Efficiency vs Threads ({args.experiments_name})")
    plt.xlabel("Number of Threads (p)")
    plt.ylabel("Efficiency (E)")
    plt.xticks(p_values)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "efficiency.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    for v, grp in df[df["p"] > 1].groupby("version"):
        plt.plot(grp["p"], grp["fps"], marker="^", linewidth=2, label=v)
    plt.title(f"Rendering Throughput vs Threads ({args.experiments_name})")
    plt.xlabel("Number of Threads (p)")
    plt.ylabel("Frames Per Second (FPS)")
    plt.xticks(p_values)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "fps_throughput.png"), dpi=300)
    plt.close()

    # Direct Comparison: Variant A (Static) vs Variant B (Shuffled)
    df_a = df[df["version"] == "Version 1 (Variant A: Static Rows)"].set_index("p")
    df_b = df[df["version"] == "Version 1 (Variant B: Shuffled Tiles)"].set_index("p")
    common_p = [p for p in p_values if p in df_a.index and p in df_b.index]

    plt.figure(figsize=(8, 6))
    plt.plot(common_p, df_a.loc[common_p, "median_time_s"], marker="o", label="Variant A: Static Rows", color="#d95f02", linewidth=2)
    plt.plot(common_p, df_b.loc[common_p, "median_time_s"], marker="s", label="Variant B: Shuffled Tiles", color="#2b5c8f", linewidth=2)
    plt.title("Load Balancing Effect: Static Rows vs Shuffled Tiles")
    plt.xlabel("Threads (p)")
    plt.ylabel("Execution Time for Benchmark Frames (s)")
    plt.xticks(common_p)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "variant_comparison.png"), dpi=300)
    plt.close()
    logger.info("Plots generated: speedup.png, efficiency.png, fps_throughput.png, variant_comparison.png")

def main():
    logger.info(f"Starting Fractal Benchmark: '{args.experiments_name}' ({args.width}x{args.height})")
    active_layer = warmup()
    save_config(active_layer)

    generate_tile_cost_heatmap()
    generate_zoom_movie()

    w, h = args.width, args.height
    tiles_arr = build_shuffled_tiles(w, h, args.tile_size)
    bench_f = args.bench_frames
    records = []

    logger.info(f"Running Baseline (Version 0: Serial) over {bench_f} benchmark frames...")
    v0_runs = []
    sample_v0 = np.zeros((h, w), dtype=np.int32)
    sample_v1_a = np.zeros((h, w), dtype=np.int32)
    sample_v1_b = np.zeros((h, w), dtype=np.int32)

    for _ in range(args.run_repetition):
        buf = np.zeros((h, w), dtype=np.int32)
        t0 = time.perf_counter()
        for f in range(bench_f):
            x0, x1, y0, y1, maxit = get_frame_bounds(f, args.frames, w, h)
            mandelbrot_serial(x0, x1, y0, y1, w, h, maxit, buf)
            if f == 0:
                sample_v0[:] = buf
        v0_runs.append(time.perf_counter() - t0)

    t_serial = float(np.median(v0_runs))
    fps_serial = bench_f / t_serial
    logger.info(f"[Baseline] Serial Time: {t_serial:.4f}s | FPS: {fps_serial:.2f}")

    records.append({
        "version": "Version 0 (Serial)",
        "p": 1,
        "median_time_s": t_serial,
        "speedup": 1.0,
        "efficiency": 1.0,
        "fps": fps_serial,
        "correct": True
    })

    for p in p_values:
        numba.set_num_threads(p)

        # Variant A: Static Rows
        runs_a = []
        for _ in range(args.run_repetition):
            buf = np.zeros((h, w), dtype=np.int32)
            t0 = time.perf_counter()
            for f in range(bench_f):
                x0, x1, y0, y1, maxit = get_frame_bounds(f, args.frames, w, h)
                mandelbrot_variant_a(x0, x1, y0, y1, w, h, maxit, buf)
                if f == 0:
                    sample_v1_a[:] = buf
            runs_a.append(time.perf_counter() - t0)
        t_a = float(np.median(runs_a))
        sp_a = t_serial / t_a
        eff_a = sp_a / p
        corr_a = bool(np.array_equal(sample_v0, sample_v1_a))

        # Variant B: Shuffled Tiles
        runs_b = []
        for _ in range(args.run_repetition):
            buf = np.zeros((h, w), dtype=np.int32)
            t0 = time.perf_counter()
            for f in range(bench_f):
                x0, x1, y0, y1, maxit = get_frame_bounds(f, args.frames, w, h)
                mandelbrot_variant_b(x0, x1, y0, y1, w, h, maxit, tiles_arr, buf)
                if f == 0:
                    sample_v1_b[:] = buf
            runs_b.append(time.perf_counter() - t0)
        t_b = float(np.median(runs_b))
        sp_b = t_serial / t_b
        eff_b = sp_b / p
        corr_b = bool(np.array_equal(sample_v0, sample_v1_b))

        logger.info(f"p={p:<3d} | Static Rows: {t_a:.4f}s ({sp_a:.2f}x) | Shuffled Tiles: {t_b:.4f}s ({sp_b:.2f}x) | Correct: {corr_a and corr_b}")

        records.append({
            "version": "Version 1 (Variant A: Static Rows)",
            "p": p,
            "median_time_s": t_a,
            "speedup": sp_a,
            "efficiency": eff_a,
            "fps": bench_f / t_a,
            "correct": corr_a
        })
        records.append({
            "version": "Version 1 (Variant B: Shuffled Tiles)",
            "p": p,
            "median_time_s": t_b,
            "speedup": sp_b,
            "efficiency": eff_b,
            "fps": bench_f / t_b,
            "correct": corr_b
        })

    df = pd.DataFrame(records)
    csv_file = os.path.join(exp_dir, "results.csv")
    df.to_csv(csv_file, index=False)
    logger.info(f"Results CSV saved to: {csv_file}")

    generate_plots(df)
    logger.info("Fractal Experiment finished successfully.")

if __name__ == "__main__":
    main()