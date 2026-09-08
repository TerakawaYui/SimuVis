import numpy as np

from config.settings import VISIBILITIES_DATA_PATH
from src.plotting import plot_reconstruction_comparison
from src.reconstruction import run_reconstruction
from src.visibility import load_sky_model


def load_visibility_results():
    """读取主流程已经保存的 UV 坐标与复可见度。"""
    if not VISIBILITIES_DATA_PATH.exists():
        raise FileNotFoundError(f"请先运行 main.py 生成可见度：{VISIBILITIES_DATA_PATH}")
    with np.load(VISIBILITIES_DATA_PATH) as data:
        uv_samples = data["uv_samples"].astype(np.float64)
        visibilities = data["visibility"].astype(np.complex128)
    return uv_samples, visibilities


def test_clean_reconstruction():
    """独立执行 dirty image 反演与 Hogbom CLEAN 实验。"""
    image, l, m, _, _ = load_sky_model()
    uv_samples, visibilities = load_visibility_results()
    reconstruction = run_reconstruction(image, l, m, uv_samples, visibilities)
    plot_reconstruction_comparison(image, reconstruction["dirty_beam"], reconstruction["dirty_image"], reconstruction["clean_image"], l, m, show_figure=False)
    assert reconstruction["clean_image"].shape == image.shape
    assert np.all(np.isfinite(reconstruction["clean_image"]))


if __name__ == "__main__":
    test_clean_reconstruction()
    print("CLEAN reconstruction experiment passed.")
