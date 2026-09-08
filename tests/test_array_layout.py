import matplotlib.pyplot as plt
import numpy as np

from config.settings import (
    ARRAY_NAME,
    NUMBER_OF_ANTENNAS,
    MINIMUM_SPACING_M,
    MAXIMUM_BASELINE_M,
    ARRAY_LAYOUT_FIGURE_PATH,
)

from src.array_geometry import (
    create_antenna_positions,
    create_baselines,
    calculate_baseline_lengths,
)


def test_array_layout():
    """
    测试并绘制理想化 ALMA Y 型阵列。
    """

    # --------------------------------------------------------
    # 1. 创建天线位置
    # --------------------------------------------------------

    positions = create_antenna_positions()

    # East 坐标
    east = positions[:, 0]

    # North 坐标
    north = positions[:, 1]

    # --------------------------------------------------------
    # 2. 计算全部 baseline
    # --------------------------------------------------------

    baselines, antenna_pairs = (
        create_baselines(positions)
    )

    baseline_lengths = (
        calculate_baseline_lengths(
            baselines
        )
    )

    # --------------------------------------------------------
    # 3. 输出基本信息
    # --------------------------------------------------------

    print("=" * 60)
    print("Array name:", ARRAY_NAME)
    print("=" * 60)

    print(
        "Number of antennas:",
        len(positions),
    )

    print(
        "Number of baselines:",
        len(baselines),
    )

    print(
        "Minimum baseline:",
        baseline_lengths.min(),
        "m",
    )

    print(
        "Maximum baseline:",
        baseline_lengths.max(),
        "m",
    )

    print(
        "Maximum baseline:",
        baseline_lengths.max() / 1000.0,
        "km",
    )

    # --------------------------------------------------------
    # 4. 自动测试
    # --------------------------------------------------------

    expected_baseline_number = (
        NUMBER_OF_ANTENNAS
        * (NUMBER_OF_ANTENNAS - 1)
        // 2
    )

    assert positions.shape == (
        NUMBER_OF_ANTENNAS,
        3,
    )

    assert len(baselines) == (
        expected_baseline_number
    )

    assert np.isclose(
        baseline_lengths.min(),
        MINIMUM_SPACING_M,
        atol=1e-6,
    )

    assert np.isclose(
        baseline_lengths.max(),
        MAXIMUM_BASELINE_M,
        rtol=1e-6,
    )

    print("All array tests passed.")

    # --------------------------------------------------------
    # 5. 创建结果文件夹
    # --------------------------------------------------------

    ARRAY_LAYOUT_FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 6. 绘制真实米制坐标
    # --------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(9, 9)
    )

    # 中央两根天线
    axis.scatter(
        east[:2],
        north[:2],
        color="red",
        marker="o",
        s=70,
        label="Core antennas",
        zorder=3,
    )

    # 三条臂上的48根天线
    axis.scatter(
        east[2:],
        north[2:],
        color="royalblue",
        marker="^",
        s=45,
        label="Y-arm antennas",
        zorder=3,
    )

    # 标出每根天线的编号
    for antenna_id in range(
        len(positions)
    ):
        axis.annotate(
            str(antenna_id),
            (
                east[antenna_id],
                north[antenna_id],
            ),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6,
        )

    axis.axhline(
        0.0,
        color="gray",
        linewidth=0.8,
        linestyle="--",
    )

    axis.axvline(
        0.0,
        color="gray",
        linewidth=0.8,
        linestyle="--",
    )

    axis.set_xlabel(
        "East coordinate / m"
    )

    axis.set_ylabel(
        "North coordinate / m"
    )

    axis.set_title(
        "ALMA-inspired Idealized Y Array"
    )

    # x、y 使用相同比例
    axis.set_aspect(
        "equal",
        adjustable="box",
    )

    axis.grid(
        True,
        linestyle=":",
        alpha=0.6,
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        ARRAY_LAYOUT_FIGURE_PATH,
        dpi=200,
    )

    print(
        "Array figure saved to:"
    )

    print(
        ARRAY_LAYOUT_FIGURE_PATH
    )

    plt.show()


if __name__ == "__main__":
    test_array_layout()