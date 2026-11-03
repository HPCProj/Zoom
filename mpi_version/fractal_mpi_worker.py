import os
import sys
import json
import time
import argparse
import numpy as np
from mpi4py import MPI
from numba import njit


@njit(parallel=False, cache=True)
def mandelbrot_tile_kernel(x0, x1, y0, y1, full_w, full_h, j_start, j_end, i_start, i_end, maxit):
    tile_h = j_end - j_start
    tile_w = i_end - i_start
    out = np.empty((tile_h, tile_w), dtype=np.int32)
    dx = (x1 - x0) / full_w
    dy = (y1 - y0) / full_h

    for j in range(tile_h):
        y = y0 + (j_start + j) * dy
        for i in range(tile_w):
            x = x0 + (i_start + i) * dx
            zr = 0.0
            zi = 0.0
            k = 0
            while k < maxit and (zr * zr + zi * zi) <= 4.0:
                zr_new = zr * zr - zi * zi + x
                zi = 2.0 * zr * zi + y
                zr = zr_new
                k += 1
            out[j, i] = k
    return out


def get_frame_bounds(frame_idx, total_frames, base_w, base_h, cx, cy, shrink_factor):
    zoom = shrink_factor ** frame_idx
    w = 3.0 * zoom
    h = w * (base_h / base_w)
    x0 = cx - w / 2.0
    x1 = cx + w / 2.0
    y0 = cy - h / 2.0
    y1 = cy + h / 2.0
    maxit = int(200 + (1000 - 200) * (frame_idx / max(1, total_frames - 1)))
    return x0, x1, y0, y1, maxit


def run_strategy_s1(comm, rank, size, width, height, frames, cx, cy, shrink):
    rows_per_rank = height // size
    start_row = rank * rows_per_rank
    end_row = start_row + rows_per_rank

    t_busy = 0.0
    t0_total = MPI.Wtime()

    all_frames_data = [] if rank == 0 else None

    for f in range(frames):
        x0, x1, y0, y1, maxit = get_frame_bounds(f, frames, width, height, cx, cy, shrink)

        tb0 = MPI.Wtime()
        local_slice = mandelbrot_tile_kernel(x0, x1, y0, y1, width, height, start_row, end_row, 0, width, maxit)
        t_busy += (MPI.Wtime() - tb0)

        recv_buf = np.empty((height, width), dtype=np.int32) if rank == 0 else None
        comm.Gather(local_slice, recv_buf, root=0)

        if rank == 0:
            all_frames_data.append(recv_buf)

    comm.Barrier()
    total_time = MPI.Wtime() - t0_total
    idle_time = max(0.0, total_time - t_busy)
    return total_time, t_busy, idle_time, all_frames_data


def run_strategy_s2(comm, rank, size, width, height, frames, cx, cy, shrink, tile_sz):
    WORK, STOP = 1, 2
    status = MPI.Status()

    nx = (width + tile_sz - 1) // tile_sz
    ny = (height + tile_sz - 1) // tile_sz

    t_busy = 0.0
    t0_total = MPI.Wtime()
    all_frames_data = [] if rank == 0 else None

    for f in range(frames):
        x0, x1, y0, y1, maxit = get_frame_bounds(f, frames, width, height, cx, cy, shrink)

        if rank == 0:
            img = np.zeros((height, width), dtype=np.int32)
            tasks = [
                (
                    j * tile_sz, min(height, (j + 1) * tile_sz),
                    i * tile_sz, min(width, (i + 1) * tile_sz)
                )
                for j in range(ny) for i in range(nx)
            ]
            nxt = 0
            active_workers = 0

            # Rank 0 is master if size > 1; otherwise handles locally
            if size == 1:
                tb0 = MPI.Wtime()
                for task in tasks:
                    tile = mandelbrot_tile_kernel(x0, x1, y0, y1, width, height, *task, maxit)
                    img[task[0]:task[1], task[2]:task[3]] = tile
                t_busy += (MPI.Wtime() - tb0)
            else:
                for w in range(1, size):
                    if nxt < len(tasks):
                        comm.send((tasks[nxt], x0, x1, y0, y1, maxit), dest=w, tag=WORK)
                        nxt += 1
                        active_workers += 1
                    else:
                        comm.send(None, dest=w, tag=STOP)

                while active_workers > 0:
                    task, tile = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                    w_source = status.Get_source()
                    img[task[0]:task[1], task[2]:task[3]] = tile

                    if nxt < len(tasks):
                        comm.send((tasks[nxt], x0, x1, y0, y1, maxit), dest=w_source, tag=WORK)
                        nxt += 1
                    else:
                        comm.send(None, dest=w_source, tag=STOP)
                        active_workers -= 1

            all_frames_data.append(img)
        else:
            while True:
                msg = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
                if status.Get_tag() == STOP:
                    break
                task, fx0, fx1, fy0, fy1, fmaxit = msg
                tb0 = MPI.Wtime()
                tile = mandelbrot_tile_kernel(fx0, fx1, fy0, fy1, width, height, *task, fmaxit)
                t_busy += (MPI.Wtime() - tb0)
                comm.send((task, tile), dest=0)

    comm.Barrier()
    total_time = MPI.Wtime() - t0_total
    idle_time = max(0.0, total_time - t_busy)
    return total_time, t_busy, idle_time, all_frames_data


def run_strategy_s3(comm, rank, size, width, height, frames, cx, cy, shrink):
    t_busy = 0.0
    t0_total = MPI.Wtime()

    local_frames = []
    local_indices = []

    for f in range(frames):
        if f % size == rank:
            x0, x1, y0, y1, maxit = get_frame_bounds(f, frames, width, height, cx, cy, shrink)
            tb0 = MPI.Wtime()
            img = mandelbrot_tile_kernel(x0, x1, y0, y1, width, height, 0, height, 0, width, maxit)
            t_busy += (MPI.Wtime() - tb0)
            local_frames.append(img)
            local_indices.append(f)

    comm.Barrier()
    gathered_indices = comm.gather(local_indices, root=0)
    gathered_frames = comm.gather(local_frames, root=0)

    all_frames_sorted = None
    if rank == 0:
        all_frames_sorted = [None] * frames
        for r in range(size):
            r_indices = gathered_indices[r]
            r_imgs = gathered_frames[r]
            for idx, img in zip(r_indices, r_imgs):
                all_frames_sorted[idx] = img

    comm.Barrier()
    total_time = MPI.Wtime() - t0_total
    idle_time = max(0.0, total_time - t_busy)
    return total_time, t_busy, idle_time, all_frames_sorted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, required=True, choices=["S1", "S2", "S3"])
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--tile_size", type=int, default=40)
    parser.add_argument("--save_movie", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # JIT Warmup
    mandelbrot_tile_kernel(0.0, 1.0, 0.0, 1.0, 16, 16, 0, 8, 0, 8, 50)
    if args.warmup:
        sys.exit(0)

    cx = -0.7436
    cy = 0.1318
    shrink = 0.95

    if args.strategy == "S1":
        total_t, busy_t, idle_t, frames_out = run_strategy_s1(
            comm, rank, size, args.width, args.height, args.frames, cx, cy, shrink
        )
    elif args.strategy == "S2":
        total_t, busy_t, idle_t, frames_out = run_strategy_s2(
            comm, rank, size, args.width, args.height, args.frames, cx, cy, shrink, args.tile_size
        )
    else:
        total_t, busy_t, idle_t, frames_out = run_strategy_s3(
            comm, rank, size, args.width, args.height, args.frames, cx, cy, shrink
        )

    all_busy = comm.gather(busy_t, root=0)
    all_idle = comm.gather(idle_t, root=0)
    max_total = comm.reduce(total_t, op=MPI.MAX, root=0)

    if rank == 0:
        out_payload = {
            "strategy": args.strategy,
            "p": size,
            "frames": args.frames,
            "total_time_s": max_total,
            "worker_busy_times": all_busy,
            "worker_idle_times": all_idle,
            "fps": args.frames / max_total
        }
        print(f"JSON_RESULT:{json.dumps(out_payload)}")

        if args.save_movie and frames_out is not None:
            import imageio.v2 as imageio
            import matplotlib.cm as cm
            colored_frames = []
            cmap = cm._colormaps["twilight_shifted"]
            for frame_mat in frames_out:
                normed = (frame_mat - frame_mat.min()) / max(1.0, (frame_mat.max() - frame_mat.min()))
                rgb = (cmap(normed)[:, :, :3] * 255).astype(np.uint8)
                colored_frames.append(rgb)
            movie_path = f"mandelbrot_zoom_{args.strategy}.gif"
            imageio.mimsave(movie_path, colored_frames, fps=24, loop=0)


if __name__ == "__main__":
    main()