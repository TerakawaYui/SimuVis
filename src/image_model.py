from PIL import Image, ImageDraw
import numpy as np

from config.settings import FIELD_OF_VIEW_DEG, IMAGE_SIZE, SKY_IMAGE_PATH, SKY_MODEL_PATH, SKY_PREVIEW_PATH


def remove_white_background(image, threshold=35):
    """把与四个角连通的白色背景填充为黑色。"""
    result = image.copy()
    width, height = result.size
    corners = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]
    for corner in corners:
        ImageDraw.floodfill(result, xy=corner, value=(0, 0, 0), thresh=threshold)
    return result


def rgb2brightness(image):
    """将 RGB 图像转换为归一化灰度天空亮度。"""
    rgb = np.asarray(image, dtype=np.float64) / 255.0
    brightness = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    brightness = np.clip(brightness, 0.0, None)
    if brightness.max() > 0:
        brightness /= brightness.max()
    return brightness


def create_direction_coord():
    """生成以弧度为单位的 l、m 坐标和像素角宽度。"""
    field_of_view_rad = np.deg2rad(FIELD_OF_VIEW_DEG)
    dl = field_of_view_rad / IMAGE_SIZE
    dm = field_of_view_rad / IMAGE_SIZE
    pixel_numbers = np.arange(IMAGE_SIZE)
    l = (pixel_numbers - (IMAGE_SIZE - 1) / 2.0) * dl
    m = -(pixel_numbers - (IMAGE_SIZE - 1) / 2.0) * dm
    return l, m, dl, dm


def create_sky_model():
    """从自然图像生成天空亮度矩阵 I(l,m)。"""
    if not SKY_IMAGE_PATH.exists():
        raise FileNotFoundError(f"找不到输入图像：{SKY_IMAGE_PATH}")

    image = Image.open(SKY_IMAGE_PATH).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    brightness = rgb2brightness(remove_white_background(image))
    l, m, dl, dm = create_direction_coord()

    SKY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(brightness * 255.0), mode="L").save(SKY_PREVIEW_PATH)
    np.savez_compressed(
        SKY_MODEL_PATH,
        I=brightness.astype(np.float32),
        l=l.astype(np.float64),
        m=m.astype(np.float64),
        dl=dl,
        dm=dm,
        field_of_view_deg=FIELD_OF_VIEW_DEG,
    )

    print(f"Sky model saved: {SKY_MODEL_PATH}")
    print(f"Sky image shape: {brightness.shape}, field of view: {FIELD_OF_VIEW_DEG * 3600:.3f} arcsec")
    return brightness, l, m, dl, dm
