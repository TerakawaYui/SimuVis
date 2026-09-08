import numpy as np

from src.plotting import plot_visibility_results
from src.visibility import (
    calculate_single_visibility,
    calculate_visibilities,
    calculate_visibilities_simple,
    load_sky_model,
    load_uv_samples,
)


def create_test_coordinates():
    size, dl, dm = 5, 1.0e-4, 1.0e-4
    l = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) * dl
    m = np.array([2.0, 1.0, 0.0, -1.0, -2.0]) * dm
    return size, l, m, dl, dm


def test_zero_baseline():
    image, l, m, dl, dm = load_sky_model()
    calculated = calculate_single_visibility(image, l, m, dl, dm, 0.0, 0.0)
    expected = np.sum(image) * dl * dm
    assert np.isclose(calculated.real, expected, rtol=1e-12, atol=1e-15)
    assert np.isclose(calculated.imag, 0.0, atol=1e-15)


def test_center_point_source():
    size, l, m, dl, dm = create_test_coordinates()
    image = np.zeros((size, size), dtype=np.float64)
    image[2, 2] = 1.0 / (dl * dm)

    for u, v in [(0.0, 0.0), (100.0, 200.0), (-500.0, 300.0), (10000.0, -8000.0)]:
        calculated = calculate_single_visibility(image, l, m, dl, dm, u, v)
        assert np.allclose(calculated, 1.0 + 0j, rtol=1e-12, atol=1e-12)


def test_offset_point_source():
    size, l, m, dl, dm = create_test_coordinates()
    image = np.zeros((size, size), dtype=np.float64)
    row, column, flux = 1, 3, 2.0
    image[row, column] = flux / (dl * dm)
    u, v = 700.0, -400.0
    calculated = calculate_single_visibility(image, l, m, dl, dm, u, v)
    expected = flux * np.exp(-2j * np.pi * (u * l[column] + v * m[row]))
    assert np.allclose(calculated, expected, rtol=1e-12, atol=1e-12)


def test_multiple_visibilities():
    image, l, m, dl, dm = load_sky_model()
    uv_samples = load_uv_samples()[:10]
    visibilities = calculate_visibilities_simple(image, l, m, dl, dm, uv_samples)
    assert visibilities.shape == (10,)
    assert np.all(np.isfinite(visibilities))


def test_fast_visibility_calculation():
    image, l, m, dl, dm = load_sky_model()
    uv_samples = load_uv_samples()[:10]
    simple_result = calculate_visibilities_simple(image, l, m, dl, dm, uv_samples)
    fast_result = calculate_visibilities(image, l, m, dl, dm, uv_samples, batch_size=4)
    difference = np.max(np.abs(simple_result - fast_result))
    print(f"Maximum fast/simple difference: {difference:.3e}")
    assert np.allclose(fast_result, simple_result, rtol=1e-10, atol=1e-12)
    plot_visibility_results(uv_samples, fast_result, show_figure=False)


def run_all_tests():
    test_zero_baseline()
    test_center_point_source()
    test_offset_point_source()
    test_multiple_visibilities()
    test_fast_visibility_calculation()
    print("All visibility tests passed.")


if __name__ == "__main__":
    run_all_tests()
