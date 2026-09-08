import numpy as np

from config.settings import (
    ANTENNAS_PER_ARM,
    HOUR_ANGLE_END_HOUR,
    HOUR_ANGLE_START_HOUR,
    MINIMUM_SPACING_M,
    NUMBER_OF_ANTENNAS,
    OBSERVATORY_LATITUDE_DEG,
    OBSERVING_FREQUENCY_HZ,
    SOURCE_DECLINATION_DEG,
    SPEED_OF_LIGHT_M_S,
    TIME_INTERVAL_MIN,
    UV_SAMPLES_DATA_PATH,
    Y_ARM_ANGLES_DEG,
    Y_ARM_INNER_RADIUS_M,
    Y_ARM_OUTER_RADIUS_M,
)


def create_antenna_positions():
    """生成 ALMA 尺度理想 Y 阵列的 ENU 坐标，单位为米。"""
    half_spacing = MINIMUM_SPACING_M / 2.0
    positions = [[-half_spacing, 0.0, 0.0], [half_spacing, 0.0, 0.0]]
    arm_radii = np.geomspace(Y_ARM_INNER_RADIUS_M, Y_ARM_OUTER_RADIUS_M, ANTENNAS_PER_ARM)

    for angle_deg in Y_ARM_ANGLES_DEG:
        angle_rad = np.deg2rad(angle_deg)
        for radius in arm_radii:
            positions.append([radius * np.cos(angle_rad), radius * np.sin(angle_rad), 0.0])

    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape != (NUMBER_OF_ANTENNAS, 3):
        raise ValueError(f"天线位置矩阵应为 {(NUMBER_OF_ANTENNAS, 3)}，实际为 {positions.shape}")
    return positions


def create_baselines(antenna_positions):
    """由天线 ENU 坐标生成所有不重复 baseline 和天线编号对。"""
    antenna_positions = np.asarray(antenna_positions, dtype=np.float64)
    baselines, antenna_pairs = [], []

    for i in range(len(antenna_positions)):
        for j in range(i + 1, len(antenna_positions)):
            baselines.append(antenna_positions[j] - antenna_positions[i])
            antenna_pairs.append([i, j])

    return np.asarray(baselines, dtype=np.float64), np.asarray(antenna_pairs, dtype=np.int32)


def calculate_baseline_lengths(baselines):
    """计算 baseline 长度，单位为米。"""
    return np.linalg.norm(baselines, axis=1)


def create_hour_angles():
    """生成 hour angle，分别返回小时和弧度。"""
    interval_hour = TIME_INTERVAL_MIN / 60.0
    hour_angles_hour = np.arange(
        HOUR_ANGLE_START_HOUR,
        HOUR_ANGLE_END_HOUR + interval_hour / 2.0,
        interval_hour,
    )
    hour_angles_rad = np.deg2rad(hour_angles_hour * 15.0)
    return hour_angles_hour, hour_angles_rad


def calculate_wavelength():
    """由观测频率计算波长，单位为米。"""
    if OBSERVING_FREQUENCY_HZ <= 0:
        raise ValueError("观测频率必须大于零。")
    return SPEED_OF_LIGHT_M_S / OBSERVING_FREQUENCY_HZ


def calculate_uvw(baselines, hour_angles_rad):
    """将 ENU baseline 转换成以波长为单位的 UVW。"""
    baselines = np.asarray(baselines, dtype=np.float64)
    hour_angles_rad = np.asarray(hour_angles_rad, dtype=np.float64)
    if baselines.ndim != 2 or baselines.shape[1] != 3:
        raise ValueError("baselines 的形状必须是 (N_baseline, 3)。")
    if hour_angles_rad.ndim != 1:
        raise ValueError("hour_angles_rad 必须是一维数组。")

    east = baselines[:, 0][None, :]
    north = baselines[:, 1][None, :]
    up = baselines[:, 2][None, :]
    hour_angle = hour_angles_rad[:, None]

    latitude = np.deg2rad(OBSERVATORY_LATITUDE_DEG)
    declination = np.deg2rad(SOURCE_DECLINATION_DEG)
    sin_h, cos_h = np.sin(hour_angle), np.cos(hour_angle)
    sin_lat, cos_lat = np.sin(latitude), np.cos(latitude)
    sin_dec, cos_dec = np.sin(declination), np.cos(declination)

    u_m = east * cos_h - north * sin_lat * sin_h + up * cos_lat * sin_h
    v_m = (
        east * sin_dec * sin_h
        + north * (sin_lat * sin_dec * cos_h + cos_lat * cos_dec)
        + up * (-cos_lat * sin_dec * cos_h + sin_lat * cos_dec)
    )
    w_m = (
        -east * cos_dec * sin_h
        + north * (-sin_lat * cos_dec * cos_h + cos_lat * sin_dec)
        + up * (cos_lat * cos_dec * cos_h + sin_lat * sin_dec)
    )

    return np.stack([u_m, v_m, w_m], axis=-1) / calculate_wavelength()


def create_uv_samples():
    """创建完整阵列、baseline、hour angle 和 UVW 数据。"""
    antenna_positions = create_antenna_positions()
    baselines, antenna_pairs = create_baselines(antenna_positions)
    hour_angles_hour, hour_angles_rad = create_hour_angles()
    uvw = calculate_uvw(baselines, hour_angles_rad)
    uv_samples = uvw[:, :, :2].reshape(-1, 2)

    return {
        "antenna_positions": antenna_positions,
        "baselines": baselines,
        "antenna_pairs": antenna_pairs,
        "hour_angles_hour": hour_angles_hour,
        "hour_angles_rad": hour_angles_rad,
        "wavelength": calculate_wavelength(),
        "uvw": uvw,
        "uv_samples": uv_samples,
    }


def save_uv_samples(data):
    """保存 UVW 和相关观测信息。"""
    UV_SAMPLES_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        UV_SAMPLES_DATA_PATH,
        uv_samples=data["uv_samples"],
        uvw=data["uvw"],
        hour_angles_hour=data["hour_angles_hour"],
        antenna_pairs=data["antenna_pairs"],
        wavelength=data["wavelength"],
    )
    print(f"UV data saved: {UV_SAMPLES_DATA_PATH}")


if __name__ == "__main__":
    uv_data = create_uv_samples()
    save_uv_samples(uv_data)
    print(f"Antennas: {uv_data['antenna_positions'].shape}")
    print(f"Baselines: {uv_data['baselines'].shape}")
    print(f"UVW: {uv_data['uvw'].shape}, UV samples: {uv_data['uv_samples'].shape}")
