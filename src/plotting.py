import matplotlib.pyplot as plt
import numpy as np

from config.settings import (
    RECONSTRUCTION_FIGURE_PATH,
    UV_COVERAGE_FIGURE_PATH,
    VISIBILITY_RESULTS_FIGURE_PATH,
)


def choose_uv_unit(u, v):
    maximum_value = max(np.max(np.abs(u)), np.max(np.abs(v)))
    if maximum_value >= 1.0e6:
        return 1.0e6, "Mlambda"
    if maximum_value >= 1.0e3:
        return 1.0e3, "klambda"
    return 1.0, "lambda"


def plot_uv_coverage(uv_samples, include_conjugates=True, show_figure=True):
    uv_samples = np.asarray(uv_samples, dtype=np.float64)
    if uv_samples.ndim != 2 or uv_samples.shape[1] != 2:
        raise ValueError("uv_samples 的形状必须是 (N_sample, 2)。")
    if not np.all(np.isfinite(uv_samples)):
        raise ValueError("UV 数据中包含 NaN 或无穷大。")

    u, v = uv_samples[:, 0], uv_samples[:, 1]
    scale, unit_label = choose_uv_unit(u, v)
    u_plot, v_plot = u / scale, v / scale
    figure, axis = plt.subplots(figsize=(9, 9))
    axis.scatter(u_plot, v_plot, s=1, color="royalblue", alpha=0.45, linewidths=0, label="Measured")
    if include_conjugates:
        axis.scatter(-u_plot, -v_plot, s=1, color="darkorange", alpha=0.30, linewidths=0, label="Conjugate")

    axis.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
    axis.axvline(0.0, color="black", linewidth=0.6, alpha=0.5)
    axis.set(xlabel=f"u / {unit_label}", ylabel=f"v / {unit_label}", title="UV Coverage")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, linestyle=":", alpha=0.4)
    axis.legend(markerscale=6)
    figure.tight_layout()

    UV_COVERAGE_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(UV_COVERAGE_FIGURE_PATH, dpi=250, bbox_inches="tight")
    print(f"UV coverage saved: {UV_COVERAGE_FIGURE_PATH}")
    if show_figure:
        plt.show()
    return figure, axis


def plot_visibility_results(uv_samples, visibilities, show_figure=True):
    uv_samples = np.asarray(uv_samples, dtype=np.float64)
    visibilities = np.asarray(visibilities, dtype=np.complex128)
    if uv_samples.shape != (len(visibilities), 2):
        raise ValueError("UV 点数量与可见度数量不一致。")
    if not np.all(np.isfinite(uv_samples)) or not np.all(np.isfinite(visibilities)):
        raise ValueError("可见度绘图数据中存在 NaN 或无穷大。")

    u, v = uv_samples[:, 0], uv_samples[:, 1]
    uv_distance = np.hypot(u, v)
    amplitude = np.abs(visibilities)
    phase_deg = np.rad2deg(np.angle(visibilities))
    maximum_amplitude = max(np.max(amplitude), np.finfo(float).eps)
    normalized_amplitude = amplitude / maximum_amplitude
    normalized_visibility = visibilities / maximum_amplitude
    scale, unit_label = choose_uv_unit(u, v)
    u_plot, v_plot, radius_plot = u / scale, v / scale, uv_distance / scale

    figure, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes[0, 0].scatter(radius_plot, normalized_amplitude, s=7, alpha=0.65, linewidths=0)
    axes[0, 0].set(xlabel=f"UV distance / {unit_label}", ylabel="Normalized amplitude", title="Amplitude")
    axes[0, 1].scatter(radius_plot, phase_deg, s=7, color="darkorange", alpha=0.65, linewidths=0)
    axes[0, 1].set(xlabel=f"UV distance / {unit_label}", ylabel="Phase / degree", title="Phase", ylim=(-180, 180))
    axes[1, 0].scatter(normalized_visibility.real, normalized_visibility.imag, s=8, color="seagreen", alpha=0.65, linewidths=0)
    axes[1, 0].set(xlabel="Normalized real", ylabel="Normalized imaginary", title="Complex Visibility")
    axes[1, 0].set_aspect("equal", adjustable="box")
    scatter = axes[1, 1].scatter(u_plot, v_plot, c=normalized_amplitude, cmap="viridis", s=8, linewidths=0)
    axes[1, 1].set(xlabel=f"u / {unit_label}", ylabel=f"v / {unit_label}", title="Amplitude on UV Plane")
    axes[1, 1].set_aspect("equal", adjustable="box")
    figure.colorbar(scatter, ax=axes[1, 1], label="Normalized amplitude")

    for axis in axes.flat:
        axis.grid(True, linestyle=":", alpha=0.4)
    figure.suptitle("Visibility Results", fontsize=16)
    figure.tight_layout()
    VISIBILITY_RESULTS_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(VISIBILITY_RESULTS_FIGURE_PATH, dpi=250, bbox_inches="tight")
    print(f"Visibility figure saved: {VISIBILITY_RESULTS_FIGURE_PATH}")
    if show_figure:
        plt.show()
    return figure, axes


def normalize_image(image, keep_sign=False):
    image = np.asarray(image, dtype=np.float64)
    scale = max(np.max(np.abs(image)), np.finfo(float).eps)
    normalized = image / scale
    return normalized if keep_sign else np.clip(normalized, 0.0, 1.0)


def plot_reconstruction_comparison(original_image, dirty_beam, dirty_image, clean_image, l, m, show_figure=True):
    """绘制原始天空、dirty beam、直接反演结果和 CLEAN 结果。"""
    original_display = normalize_image(original_image)
    beam_display = normalize_image(dirty_beam, keep_sign=True)
    dirty_display = normalize_image(dirty_image)
    clean_display = normalize_image(clean_image)
    l_arcsec = np.rad2deg(l) * 3600.0
    m_arcsec = np.rad2deg(m) * 3600.0
    extent = [l_arcsec[0], l_arcsec[-1], m_arcsec[-1], m_arcsec[0]]

    figure, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.ravel()
    original_plot = axes[0].imshow(original_display, origin="upper", extent=extent, cmap="inferno", vmin=0, vmax=1)
    beam_plot = axes[1].imshow(beam_display, origin="upper", extent=extent, cmap="RdBu_r", vmin=-1, vmax=1)
    dirty_plot = axes[2].imshow(dirty_display, origin="upper", extent=extent, cmap="inferno", vmin=0, vmax=1)
    clean_plot = axes[3].imshow(clean_display, origin="upper", extent=extent, cmap="inferno", vmin=0, vmax=1)
    axes[0].set_title("Original Sky")
    axes[1].set_title("Dirty Beam")
    axes[2].set_title("Dirty Image from Visibilities")
    axes[3].set_title("Hogbom CLEAN Image")

    for axis in axes:
        axis.set_xlabel("l / arcsec")
        axis.set_ylabel("m / arcsec")
    figure.colorbar(original_plot, ax=axes[0], fraction=0.046, pad=0.04)
    figure.colorbar(beam_plot, ax=axes[1], fraction=0.046, pad=0.04)
    figure.colorbar(dirty_plot, ax=axes[2], fraction=0.046, pad=0.04)
    figure.colorbar(clean_plot, ax=axes[3], fraction=0.046, pad=0.04)
    figure.tight_layout()

    RECONSTRUCTION_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(RECONSTRUCTION_FIGURE_PATH, dpi=250, bbox_inches="tight")
    print(f"Reconstruction figure saved: {RECONSTRUCTION_FIGURE_PATH}")
    if show_figure:
        plt.show()
    return figure, axes
