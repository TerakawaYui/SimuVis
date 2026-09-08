import time

import numpy as np

from config.settings import (
    SKY_MODEL_PATH,
    UV_SAMPLES_DATA_PATH,
    VISIBILITIES_DATA_PATH,
    VISIBILITY_BATCH_SIZE,
    VISIBILITY_SAMPLE_LIMIT,
)


def load_sky_model():
    """读取 I(l,m)、方向坐标和像素角宽度。"""
    if not SKY_MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到天空模型：{SKY_MODEL_PATH}")
    with np.load(SKY_MODEL_PATH) as data:
        image = data["I"].astype(np.float64)
        l = data["l"].astype(np.float64)
        m = data["m"].astype(np.float64)
        dl, dm = float(data["dl"]), float(data["dm"])
    return image, l, m, dl, dm


def load_uv_samples():
    """读取形状为 (N_sample, 2) 的 UV 采样坐标。"""
    if not UV_SAMPLES_DATA_PATH.exists():
        raise FileNotFoundError(f"找不到 UV 数据：{UV_SAMPLES_DATA_PATH}")
    with np.load(UV_SAMPLES_DATA_PATH) as data:
        return data["uv_samples"].astype(np.float64)


def validate_visibility_inputs(image, l, m, dl, dm, uv_samples):
    """检查天空模型和 UV 数据。"""
    if image.ndim != 2:
        raise ValueError("image 必须是二维数组。")
    if l.ndim != 1 or m.ndim != 1:
        raise ValueError("l 和 m 必须是一维数组。")
    if image.shape != (len(m), len(l)):
        raise ValueError(f"image 应为 {(len(m), len(l))}，实际为 {image.shape}。")
    if uv_samples.ndim != 2 or uv_samples.shape[1] != 2:
        raise ValueError("uv_samples 的形状必须是 (N_sample, 2)。")
    if dl <= 0 or dm <= 0:
        raise ValueError("dl 和 dm 必须大于零。")
    if not np.all(np.isfinite(image)) or not np.all(np.isfinite(uv_samples)):
        raise ValueError("输入数据中存在 NaN 或无穷大。")


def calculate_single_visibility(image, l, m, dl, dm, u, v):
    """用二维相位矩阵直接计算一个 (u,v) 点的复可见度。"""
    image = np.asarray(image, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    validate_visibility_inputs(image, l, m, dl, dm, np.array([[u, v]], dtype=np.float64))
    phase = np.exp(-2j * np.pi * (u * l[None, :] + v * m[:, None]))
    return np.sum(image * phase) * dl * dm


def calculate_visibilities_simple(image, l, m, dl, dm, uv_samples):
    """逐点计算可见度，仅用于正确性对照。"""
    validate_visibility_inputs(image, l, m, dl, dm, uv_samples)
    visibilities = np.empty(len(uv_samples), dtype=np.complex128)
    for index, (u, v) in enumerate(uv_samples):
        visibilities[index] = calculate_single_visibility(image, l, m, dl, dm, u, v)
    return visibilities


def calculate_visibilities(image, l, m, dl, dm, uv_samples, batch_size=VISIBILITY_BATCH_SIZE):
    """分批执行直接非均匀 DFT；没有使用 FFT。"""
    image = np.asarray(image, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    uv_samples = np.asarray(uv_samples, dtype=np.float64)
    validate_visibility_inputs(image, l, m, dl, dm, uv_samples)
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于零。")

    visibilities = np.empty(len(uv_samples), dtype=np.complex128)
    for start in range(0, len(uv_samples), batch_size):
        end = min(start + batch_size, len(uv_samples))
        u_batch, v_batch = uv_samples[start:end, 0], uv_samples[start:end, 1]
        phase_l = np.exp(-2j * np.pi * np.outer(l, u_batch))
        phase_m = np.exp(-2j * np.pi * np.outer(m, v_batch))
        visibilities[start:end] = np.sum(phase_m * (image @ phase_l), axis=0) * dl * dm
        print(f"\rVisibility: {end}/{len(uv_samples)}", end="", flush=True)
    print()
    return visibilities


def select_uv_samples(uv_samples, sample_limit=VISIBILITY_SAMPLE_LIMIT):
    """从完整 UV coverage 中均匀抽取样本；sample_limit=None 表示全部样本。"""
    uv_samples = np.asarray(uv_samples, dtype=np.float64)
    if sample_limit is None or sample_limit >= len(uv_samples):
        indices = np.arange(len(uv_samples))
    elif sample_limit <= 0:
        raise ValueError("sample_limit 必须大于零或为 None。")
    else:
        indices = np.linspace(0, len(uv_samples) - 1, sample_limit, dtype=np.int64)
    return uv_samples[indices], indices


def save_visibility_results(uv_samples, visibilities, sample_indices):
    """保存 UV 坐标、复可见度、振幅和相位。"""
    VISIBILITIES_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        VISIBILITIES_DATA_PATH,
        u=uv_samples[:, 0],
        v=uv_samples[:, 1],
        uv_samples=uv_samples,
        visibility=visibilities,
        visibility_real=visibilities.real,
        visibility_imag=visibilities.imag,
        amplitude=np.abs(visibilities),
        phase_rad=np.angle(visibilities),
        phase_deg=np.rad2deg(np.angle(visibilities)),
        sample_indices=sample_indices,
    )
    print(f"Visibility data saved: {VISIBILITIES_DATA_PATH}")


def run_visibility_calculation():
    """读取现有天空与 UV 数据，计算并保存选中的可见度。"""
    image, l, m, dl, dm = load_sky_model()
    uv_samples, sample_indices = select_uv_samples(load_uv_samples())
    start_time = time.perf_counter()
    visibilities = calculate_visibilities(image, l, m, dl, dm, uv_samples)
    elapsed_time = time.perf_counter() - start_time
    save_visibility_results(uv_samples, visibilities, sample_indices)
    print(f"Calculated {len(visibilities)} visibilities in {elapsed_time:.2f} s")
    return uv_samples, visibilities


if __name__ == "__main__":
    run_visibility_calculation()
