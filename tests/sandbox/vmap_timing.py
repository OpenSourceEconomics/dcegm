import math
import time

import jax
import jax.numpy as jnp

N_STATES = 5_000
N_DECISIONS = 200
N_RUNS = 100


def simple_op(x):
    return x * x + 2.0 * x


vmap_2d = jax.jit(jax.vmap(jax.vmap(simple_op, in_axes=0), in_axes=0))
vmap_3d = jax.jit(
    jax.vmap(
        jax.vmap(
            jax.vmap(simple_op, in_axes=0),
            in_axes=0,
        ),
        in_axes=0,
    )
)


def time_function(func, x, n_runs):
    func(x).block_until_ready()
    run_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(x).block_until_ready()
        end = time.perf_counter()
        run_times.append(end - start)

    avg = sum(run_times) / n_runs
    if n_runs > 1:
        variance = sum((t - avg) ** 2 for t in run_times) / (n_runs - 1)
        se = math.sqrt(variance) / math.sqrt(n_runs)
    else:
        se = 0.0
    return avg, se


def main():
    x_2d = jnp.arange(N_STATES * N_DECISIONS, dtype=jnp.float32).reshape(
        N_STATES,
        N_DECISIONS,
    )
    x_3d = x_2d[:, None, :]

    avg_2d, se_2d = time_function(vmap_2d, x_2d, N_RUNS)
    avg_3d, se_3d = time_function(vmap_3d, x_3d, N_RUNS)

    y_2d = vmap_2d(x_2d)
    y_3d = vmap_3d(x_3d).squeeze(axis=1)
    max_diff = jnp.max(jnp.abs(y_2d - y_3d))

    print(f"Input 2D shape: {x_2d.shape}")
    print(f"Input 3D shape: {x_3d.shape}")
    print(f"Average time (2D vmap): {avg_2d * 1e3:.3f} ms")
    print(f"SE time (2D vmap): {se_2d * 1e3:.3f} ms")
    print(f"Average time (3D with singleton axis): {avg_3d * 1e3:.3f} ms")
    print(f"SE time (3D with singleton axis): {se_3d * 1e3:.3f} ms")
    print(f"Max absolute difference in outputs: {float(max_diff):.6f}")


if __name__ == "__main__":
    main()
