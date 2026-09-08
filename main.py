from src.array_geometry import create_uv_samples, save_uv_samples
from src.image_model import create_sky_model
from src.plotting import plot_uv_coverage, plot_visibility_results
from src.visibility import calculate_visibilities, save_visibility_results, select_uv_samples


def main():
    print("=" * 60)
    print("Radio Interferometer Simulation")
    print("=" * 60)

    image, l, m, dl, dm = create_sky_model()

    uv_data = create_uv_samples()
    save_uv_samples(uv_data)
    plot_uv_coverage(uv_data["uv_samples"], show_figure=False)

    uv_samples, sample_indices = select_uv_samples(uv_data["uv_samples"])
    visibilities = calculate_visibilities(image, l, m, dl, dm, uv_samples)
    save_visibility_results(uv_samples, visibilities, sample_indices)
    plot_visibility_results(uv_samples, visibilities, show_figure=False)

    print("=" * 60)
    print("Visibility simulation finished. See results/data and results/figures.")
    print("=" * 60)


if __name__ == "__main__":
    main()
