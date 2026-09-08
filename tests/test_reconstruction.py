import numpy as np

from src.reconstruction import hogbom_clean, reconstruct_dirty_image
from src.visibility import calculate_visibilities


def test_center_point_reconstruction():
    """中心点源的 dirty image 应与 dirty beam 一致，且峰值位于中心。"""
    size = 33
    dl = dm = 1.0e-5
    l = (np.arange(size) - size // 2) * dl
    m = -(np.arange(size) - size // 2) * dm
    image = np.zeros((size, size), dtype=np.float64)
    image[size // 2, size // 2] = 1.0 / (dl * dm)

    rng = np.random.default_rng(42)
    uv_samples = rng.uniform(-30000.0, 30000.0, size=(64, 2))
    visibilities = calculate_visibilities(image, l, m, dl, dm, uv_samples, batch_size=16)
    dirty_image, dirty_beam, imaginary_residual = reconstruct_dirty_image(uv_samples, visibilities, l, m)

    peak_index = np.unravel_index(np.argmax(dirty_image), dirty_image.shape)
    assert peak_index == (size // 2, size // 2)
    assert np.allclose(dirty_image, dirty_beam, rtol=1e-12, atol=1e-12)
    assert np.max(np.abs(imaginary_residual)) < 1e-12


def test_hogbom_clean_center_point():
    """中心点 dirty beam 经过 CLEAN 后应留下中心 clean component。"""
    size = 33
    y, x = np.indices((size, size))
    center = size // 2
    dirty_beam = np.exp(-((x - center) ** 2 + (y - center) ** 2) / 8.0)
    clean_image, components, residual, clean_beam = hogbom_clean(dirty_beam, dirty_beam, gain=0.2, threshold_fraction=1.0e-3, max_iterations=100, positive_only=True)
    assert np.unravel_index(np.argmax(components), components.shape) == (center, center)
    assert np.max(np.abs(residual)) < 1.0e-3
    assert np.unravel_index(np.argmax(clean_image), clean_image.shape) == (center, center)
    assert np.isclose(clean_beam[center, center], 1.0)


if __name__ == "__main__":
    test_center_point_reconstruction()
    test_hogbom_clean_center_point()
    print("Reconstruction test passed.")
