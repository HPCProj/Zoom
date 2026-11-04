import os
import sys
import json
import logging
import argparse
import platform
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel Fractal Zoom Movie MPI Benchmark")
    parser.add_argument("--experiments_name", "-experiments_name", type=str, required=True)
    parser.add_argument("--results_folder", "-results_folder", type=str, default="results")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--p_list", "-p_list", type=str, default="1,2,4,8,16,32")
    parser.add_argument("--tile_size", type=int, default=40)
    parser.add_argument("--run_repetition", "-run_repetition", type=int, default=5)
    return parser.parse_args()


args = parse_args()
p_values = [int(p.strip()) for p in args.p_list.split(",") if p.strip()]

exp_dir = os.path.join(args.results_folder, args.experiments_name)
os.makedirs(exp_dir, exist_ok=True)


def setup_logger():
    logger = logging.getLogger("MPI_Fractal_Driver")
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


def run_mpi_command(strategy, p, width, height, frames, tile_sz, save_movie=False):
    cmd = [
        "mpirun", "--oversubscribe", "-n", str(p),
        sys.executable, "fractal_mpi_worker.py",
        "--strategy", strategy,
        "--width", str(width),
        "--height", str(height),
        "--frames", str(frames),
        "--tile_size", str(tile_sz)
    ]
    if save_movie:
        cmd.append("--save_movie")

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        logger.error(f"mpirun failed ({strategy}, p={p}): {res.stderr.strip()}")
        return None

    for line in res.stdout.splitlines():
        if line.startswith("JSON_RESULT:"):
            return json.loads(line.replace("JSON_RESULT:", ""))
    return None


def generate_load_balance_plot(strategy_records, p_target):
    plt.figure(figsize=(10, 5.5))
    strategies = ["S1", "S2", "S3"]
    bar_width = 0.25
    x = np.arange(p_target)

    for idx, strat in enumerate(strategies):
        matched = [r for r in strategy_records if r["strategy"] == strat and r["p"] == p_target]
        if not matched:
            continue
        data = matched[0]
        busy = data["worker_busy_times"]
        idle = data["worker_idle_times"]
        pos = x + (idx - 1) * bar_width

        plt.bar(pos, busy, width=bar_width, label=f"{strat} Busy (Compute)")
        plt.bar(pos, idle, width=bar_width, bottom=busy, alpha=0.35, hatch="//", label=f"{strat} Idle (Wait)")

    plt.title(f"Load Balance: Per-Worker Busy vs. Idle Time (p = {p_target} Workers)")
    plt.xlabel("Worker Rank Index")
    plt.ylabel("Time (seconds)")
    plt.xticks(x, [f"Rank {i}" for i in range(p_target)])
    plt.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.legend(ncol=3, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "worker_load_balance.png"), dpi=300)
    plt.close()


def generate_plots(df):
    logical_cores = os.cpu_count() or 1

    # 1. Speedup Plot
    plt.figure(figsize=(8, 6))
    for strat, grp in df.groupby("strategy"):
        plt.plot(grp["p"], grp["speedup"], marker="o", linewidth=2, label=f"Strategy {strat}")
    plt.axvline(x=logical_cores, color="red", linestyle=":", alpha=0.7, label=f"Cores ({logical_cores})")
    plt.title(f"MPI Fractal Zoom Speedup vs Processes ({args.experiments_name})")
    plt.xlabel("Processes (p)")
    plt.ylabel("Speedup (S = T1 / Tp)")
    plt.xticks(p_values)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "speedup.png"), dpi=300)
    plt.close()

    # 2. Frames Per Second (FPS) Plot
    plt.figure(figsize=(8, 6))
    for strat, grp in df.groupby("strategy"):
        plt.plot(grp["p"], grp["fps"], marker="s", linewidth=2, label=f"Strategy {strat}")
    plt.title(f"Render Throughput (Frames per Second) ({args.experiments_name})")
    plt.xlabel("Processes (p)")
    plt.ylabel("Throughput (FPS)")
    plt.xticks(p_values)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "fps_scaling.png"), dpi=300)
    plt.close()

    logger.info("Saved plots: speedup.png, fps_scaling.png, worker_load_balance.png")


def main():
    logger.info(f"Starting MPI Fractal Zoom Benchmark '{args.experiments_name}'")

    config = {
        "experiment_name": args.experiments_name,
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "p_list": p_values,
        "tile_size": args.tile_size,
        "run_repetition": args.run_repetition,
        "system": {
            "os": platform.system(),
            "cpu_cores": os.cpu_count(),
            "python": platform.python_version()
        }
    }
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    logger.info("Configuration saved to config.json")

    # MPI Warmup
    subprocess.run(["mpirun", "-n", "1", sys.executable, "fractal_mpi_worker.py", "--strategy", "S1", "--warmup"])

    strategies = ["S1", "S2", "S3"]
    records = []
    load_balance_records = []

    # Get Single-process baseline (p=1) using S1
    logger.info("Calculating baseline time for p=1...")
    res_base = run_mpi_command("S1", 1, args.width, args.height, args.frames, args.tile_size)
    t_serial = res_base["total_time_s"] if res_base else 1.0
    logger.info(f"Baseline Serial Execution Time: {t_serial:.4f}s")

    for strat in strategies:
        for p in p_values:
            if args.height % p != 0 and strat == "S1":
                logger.warning(f"Skipping S1 for p={p}: Height {args.height} not divisible by {p}.")
                continue

            runs = []
            for r in range(args.run_repetition):
                # Save GIF movie once on the last iteration of S3
                save_mov = (strat == "S3" and p == max(p_values) and r == 0)
                res = run_mpi_command(strat, p, args.width, args.height, args.frames, args.tile_size, save_mov)
                if res:
                    runs.append(res)

            if not runs:
                continue

            med_time = float(np.median([rn["total_time_s"] for rn in runs]))
            med_fps = float(np.median([rn["fps"] for rn in runs]))
            speedup = t_serial / med_time
            efficiency = speedup / p

            # Save per-worker busy times for load balance analysis
            best_run = runs[0]
            if p == max(p_values):
                load_balance_records.append({
                    "strategy": strat,
                    "p": p,
                    "worker_busy_times": best_run["worker_busy_times"],
                    "worker_idle_times": best_run["worker_idle_times"]
                })

            logger.info(
                f"[{strat}] p={p:<2d} | Median Time: {med_time:.4f}s | "
                f"FPS: {med_fps:.2f} | Speedup: {speedup:.2f}x | Efficiency: {efficiency:.2f}"
            )

            records.append({
                "strategy": strat,
                "p": p,
                "median_time_s": round(med_time, 4),
                "speedup": round(speedup, 4),
                "efficiency": round(efficiency, 4),
                "fps": round(med_fps, 2)
            })

    df = pd.DataFrame(records)
    csv_path = os.path.join(exp_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Results CSV stored at: {csv_path}")

    # Generate Plots
    generate_load_balance_plot(load_balance_records, max(p_values))
    generate_plots(df)

    # Move movie to experiment folder if generated
    if os.path.exists("mandelbrot_zoom_S3.gif"):
        os.replace("mandelbrot_zoom_S3.gif", os.path.join(exp_dir, "fractal_zoom.gif"))
        logger.info(f"Fractal Zoom Movie generated at: {os.path.join(exp_dir, 'fractal_zoom.gif')}")

    logger.info("Fractal MPI Experiment finished successfully.")


if __name__ == "__main__":
    main()