import numpy as np

from config.settings import (
    NUMBER_OF_ANTENNAS,
    UV_SAMPLES_DATA_PATH,
)

from src.array_geometry import (
    create_uv_samples,
)

from src.plotting import (
    plot_uv_coverage,
)


def test_uv_coverage():
    """
    测试 UV 数据并绘制 UV coverage。
    """

    # --------------------------------------------------------
    # 1. 生成 UV 数据
    # --------------------------------------------------------

    data = create_uv_samples()

    baselines = data["baselines"]

    hour_angles_hour = (
        data["hour_angles_hour"]
    )

    uvw = data["uvw"]

    uv_samples = data["uv_samples"]

    # --------------------------------------------------------
    # 2. 计算预期数据量
    # --------------------------------------------------------

    expected_baseline_number = (
        NUMBER_OF_ANTENNAS
        * (NUMBER_OF_ANTENNAS - 1)
        // 2
    )

    expected_time_number = len(
        hour_angles_hour
    )

    expected_uv_number = (
        expected_baseline_number
        * expected_time_number
    )

    # --------------------------------------------------------
    # 3. 检查数组形状
    # --------------------------------------------------------

    assert baselines.shape == (
        expected_baseline_number,
        3,
    )

    assert uvw.shape == (
        expected_time_number,
        expected_baseline_number,
        3,
    )

    assert uv_samples.shape == (
        expected_uv_number,
        2,
    )

    # --------------------------------------------------------
    # 4. 检查非法数据
    # --------------------------------------------------------

    assert np.all(
        np.isfinite(uvw)
    )

    assert np.all(
        np.isfinite(uv_samples)
    )

    # --------------------------------------------------------
    # 5. 输出信息
    # --------------------------------------------------------

    print("=" * 60)
    print("UV coverage test")
    print("=" * 60)

    print(
        "Number of antennas:",
        NUMBER_OF_ANTENNAS,
    )

    print(
        "Number of baselines:",
        len(baselines),
    )

    print(
        "Number of time samples:",
        len(hour_angles_hour),
    )

    print(
        "UVW shape:",
        uvw.shape,
    )

    print(
        "UV samples shape:",
        uv_samples.shape,
    )

    print(
        "Wavelength:",
        data["wavelength"],
        "m",
    )

    maximum_uv_distance = np.max(
        np.sqrt(
            uv_samples[:, 0] ** 2
            + uv_samples[:, 1] ** 2
        )
    )

    print(
        "Maximum projected UV distance:",
        maximum_uv_distance,
        "lambda",
    )

    print(
        "Maximum projected UV distance:",
        maximum_uv_distance / 1.0e6,
        "Mlambda",
    )

    print("All UV tests passed.")

    # --------------------------------------------------------
    # 6. 保存 UV 数据
    # --------------------------------------------------------

    UV_SAMPLES_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        UV_SAMPLES_DATA_PATH,
        uv_samples=uv_samples,
        uvw=uvw,
        hour_angles_hour=(
            hour_angles_hour
        ),
        antenna_pairs=(
            data["antenna_pairs"]
        ),
        wavelength=(
            data["wavelength"]
        ),
    )

    print(
        "UV data saved to:"
    )

    print(
        UV_SAMPLES_DATA_PATH
    )

    # --------------------------------------------------------
    # 7. 绘制 UV coverage
    # --------------------------------------------------------

    plot_uv_coverage(
        uv_samples,
        include_conjugates=True,
        show_figure=True,
    )


if __name__ == "__main__":
    test_uv_coverage()