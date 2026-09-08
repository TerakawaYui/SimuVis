import numpy as np

from config.settings import (
    CLEAN_GAIN,
    CLEAN_MAX_ITERATIONS,
    CLEAN_POSITIVE_ONLY,
    CLEAN_THRESHOLD_FRACTION,
    RECONSTRUCTION_BATCH_SIZE,
    RECONSTRUCTION_DATA_PATH,
)


def add_conjugate_samples(uv_samples, visibilities):
    """利用实值天空的 Hermitian 对称性补充 (-u,-v,V*)。"""
    uv_samples = np.asarray(uv_samples, dtype=np.float64)
    visibilities = np.asarray(visibilities, dtype=np.complex128)
    if uv_samples.shape != (len(visibilities), 2):
        raise ValueError("uv_samples 与 visibilities 的形状不匹配。")
    uv_full = np.concatenate([uv_samples, -uv_samples], axis=0)
    visibility_full = np.concatenate([visibilities, np.conjugate(visibilities)])
    return uv_full, visibility_full


def adjoint_nudft(uv_samples, values, l, m, batch_size=RECONSTRUCTION_BATCH_SIZE):
    """计算 sum_k values_k exp[+2 pi i (u_k l + v_k m)] / N。"""
    uv_samples = np.asarray(uv_samples, dtype=np.float64)
    values = np.asarray(values, dtype=np.complex128)
    l = np.asarray(l, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)

    if uv_samples.shape != (len(values), 2):
        raise ValueError("uv_samples 与 values 的形状不匹配。")
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于零。")

    output = np.zeros((len(m), len(l)), dtype=np.complex128)
    for start in range(0, len(values), batch_size):
        end = min(start + batch_size, len(values))
        u_batch, v_batch = uv_samples[start:end, 0], uv_samples[start:end, 1]
        phase_l = np.exp(2j * np.pi * np.outer(u_batch, l))
        phase_m = np.exp(2j * np.pi * np.outer(v_batch, m))
        output += phase_m.T @ (values[start:end, None] * phase_l)
        print(f"\rInverse DFT: {end}/{len(values)}", end="", flush=True)
    print()
    return output / len(values)


def reconstruct_dirty_image(uv_samples, visibilities, l, m):
    """由非均匀可见度重建 dirty image，并计算对应 dirty beam。"""
    uv_full, visibility_full = add_conjugate_samples(uv_samples, visibilities)
    dirty_complex = adjoint_nudft(uv_full, visibility_full, l, m)
    dirty_beam_complex = adjoint_nudft(uv_full, np.ones(len(uv_full)), l, m)
    return dirty_complex.real, dirty_beam_complex.real, dirty_complex.imag


def subtract_shifted_beam(residual, dirty_beam, component, position, beam_peak_position):
    """从 residual 中减去平移至 position 的 dirty beam。"""
    height, width = residual.shape
    row_shift = position[0] - beam_peak_position[0]
    column_shift = position[1] - beam_peak_position[1]
    residual_row_start = max(0, row_shift)
    residual_row_end = min(height, height + row_shift)
    residual_column_start = max(0, column_shift)
    residual_column_end = min(width, width + column_shift)
    beam_row_start = max(0, -row_shift)
    beam_row_end = beam_row_start + residual_row_end - residual_row_start
    beam_column_start = max(0, -column_shift)
    beam_column_end = beam_column_start + residual_column_end - residual_column_start
    residual[residual_row_start:residual_row_end, residual_column_start:residual_column_end] -= component * dirty_beam[beam_row_start:beam_row_end, beam_column_start:beam_column_end]


def estimate_clean_beam(dirty_beam):
    """用 dirty beam 中央主瓣的半高宽构造椭圆高斯 clean beam。"""
    dirty_beam = np.asarray(dirty_beam, dtype=np.float64)
    peak_position = np.unravel_index(np.argmax(np.abs(dirty_beam)), dirty_beam.shape)
    normalized_beam = dirty_beam / dirty_beam[peak_position]

    def half_max_width(profile, center):
        left = center
        right = center
        while left > 0 and profile[left - 1] >= 0.5:
            left -= 1
        while right < len(profile) - 1 and profile[right + 1] >= 0.5:
            right += 1
        return max(float(right - left + 1), 1.0)

    fwhm_x = half_max_width(normalized_beam[peak_position[0], :], peak_position[1])
    fwhm_y = half_max_width(normalized_beam[:, peak_position[1]], peak_position[0])
    sigma_x = fwhm_x / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_y = fwhm_y / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    y, x = np.indices(dirty_beam.shape)
    clean_beam = np.exp(-0.5 * (((x - peak_position[1]) / sigma_x) ** 2 + ((y - peak_position[0]) / sigma_y) ** 2))
    return clean_beam


def hogbom_clean(dirty_image, dirty_beam, gain=CLEAN_GAIN, threshold_fraction=CLEAN_THRESHOLD_FRACTION, max_iterations=CLEAN_MAX_ITERATIONS, positive_only=CLEAN_POSITIVE_ONLY):
    """执行基础 Hogbom CLEAN，返回恢复图、分量模型、残差和 clean beam。"""
    dirty_image = np.asarray(dirty_image, dtype=np.float64)
    dirty_beam = np.asarray(dirty_beam, dtype=np.float64)
    if dirty_image.shape != dirty_beam.shape:
        raise ValueError("dirty_image 与 dirty_beam 的形状必须相同。")
    if not 0.0 < gain <= 1.0:
        raise ValueError("CLEAN gain 必须在 (0, 1] 内。")
    if not 0.0 <= threshold_fraction < 1.0:
        raise ValueError("CLEAN threshold_fraction 必须在 [0, 1) 内。")
    if max_iterations <= 0:
        raise ValueError("CLEAN max_iterations 必须大于零。")

    beam_peak_position = np.unravel_index(np.argmax(np.abs(dirty_beam)), dirty_beam.shape)
    beam_peak = dirty_beam[beam_peak_position]
    if np.isclose(beam_peak, 0.0):
        raise ValueError("dirty beam 峰值为零，无法执行 CLEAN。")
    normalized_beam = dirty_beam / beam_peak
    residual = dirty_image / beam_peak
    clean_components = np.zeros_like(residual)
    threshold = threshold_fraction * np.max(np.abs(residual))

    completed_iterations = 0
    for iteration in range(max_iterations):
        peak_position = np.unravel_index(np.argmax(residual if positive_only else np.abs(residual)), residual.shape)
        peak_value = residual[peak_position]
        if (positive_only and peak_value <= threshold) or (not positive_only and abs(peak_value) <= threshold):
            break
        component = gain * peak_value
        clean_components[peak_position] += component
        subtract_shifted_beam(residual, normalized_beam, component, peak_position, beam_peak_position)
        completed_iterations = iteration + 1

    clean_beam = estimate_clean_beam(normalized_beam)
    restoring_kernel = np.fft.ifftshift(clean_beam)
    restored_components = np.fft.ifft2(np.fft.fft2(clean_components) * np.fft.fft2(restoring_kernel)).real
    clean_image = restored_components + residual
    print(f"CLEAN iterations: {completed_iterations}/{max_iterations}, residual peak: {np.max(np.abs(residual)):.3e}")
    return clean_image, clean_components, residual, clean_beam


def save_reconstruction(original_image, dirty_image, dirty_beam, imaginary_residual, clean_image, clean_components, clean_residual, clean_beam, l, m):
    """保存直接反演结果与 CLEAN 重建结果。"""
    RECONSTRUCTION_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        RECONSTRUCTION_DATA_PATH,
        original_image=original_image,
        dirty_image=dirty_image,
        dirty_beam=dirty_beam,
        imaginary_residual=imaginary_residual,
        clean_image=clean_image,
        clean_components=clean_components,
        clean_residual=clean_residual,
        clean_beam=clean_beam,
        l=l,
        m=m,
    )
    print(f"Reconstruction data saved: {RECONSTRUCTION_DATA_PATH}")


def run_reconstruction(original_image, l, m, uv_samples, visibilities):
    """完成反演、保存数据并返回结果字典。"""
    dirty_image, dirty_beam, imaginary_residual = reconstruct_dirty_image(
        uv_samples, visibilities, l, m
    )
    clean_image, clean_components, clean_residual, clean_beam = hogbom_clean(dirty_image, dirty_beam)
    save_reconstruction(original_image, dirty_image, dirty_beam, imaginary_residual, clean_image, clean_components, clean_residual, clean_beam, l, m)
    peak = max(np.max(np.abs(dirty_image)), np.finfo(float).eps)
    leakage = np.max(np.abs(imaginary_residual)) / peak
    print(f"Maximum imaginary leakage: {leakage:.3e}")
    return {
        "original_image": original_image,
        "dirty_image": dirty_image,
        "dirty_beam": dirty_beam,
        "imaginary_residual": imaginary_residual,
        "imaginary_leakage": leakage,
        "clean_image": clean_image,
        "clean_components": clean_components,
        "clean_residual": clean_residual,
        "clean_beam": clean_beam,
    }
