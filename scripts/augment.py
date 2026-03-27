"""
Image augmentation utilities shared across the mapclass pipeline.

Augmentations are designed to bridge the visual gap between clean hex tile
art and the aged, faded, monochrome appearance of real historical maps.

All functions accept and return PIL Images (RGB or RGBA) and are stateless
— no randomness unless the function name starts with 'random_'.
"""

import numpy as np
from PIL import Image, ImageFilter


# ---------------------------------------------------------------------------
# Background compositing
# ---------------------------------------------------------------------------

# Named background colours used across augmentations
BG_WHITE = (255, 255, 255)
BG_PARCHMENT = (220, 200, 165)   # warm beige, typical aged paper
BG_DARK = (30, 25, 20)           # near-black for dark-map styles


def composite(img: Image.Image, bg: tuple[int, int, int] = BG_WHITE) -> Image.Image:
    """
    Composite an RGBA image onto a solid background colour, returning RGB.
    If the image is already RGB, returns it unchanged.
    """
    if img.mode == "RGB":
        return img
    bg_img = Image.new("RGB", img.size, bg)
    if img.mode == "RGBA":
        bg_img.paste(img, mask=img.split()[3])
    else:
        bg_img.paste(img.convert("RGB"))
    return bg_img


# ---------------------------------------------------------------------------
# Style augmentations
# ---------------------------------------------------------------------------

def to_grayscale(img: Image.Image, bg: tuple[int, int, int] = BG_WHITE) -> Image.Image:
    """Convert to grayscale (returned as 3-channel RGB for model compatibility)."""
    rgb = composite(img, bg)
    return rgb.convert("L").convert("RGB")


def to_parchment(img: Image.Image) -> Image.Image:
    """
    Apply an aged parchment effect:
      1. Composite on parchment-coloured background
      2. Sepia tone via matrix transform
      3. Warm yellow-brown tint to simulate paper aging
      4. Low-amplitude Gaussian noise (paper grain)
      5. Slight blur (loss of sharpness over centuries)
    """
    rgb = composite(img, BG_PARCHMENT)
    arr = np.array(rgb, dtype=np.float32)

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Standard sepia matrix
    sr = np.clip(r * 0.393 + g * 0.769 + b * 0.189, 0, 255)
    sg = np.clip(r * 0.349 + g * 0.686 + b * 0.168, 0, 255)
    sb = np.clip(r * 0.272 + g * 0.534 + b * 0.131, 0, 255)
    arr = np.stack([sr, sg, sb], axis=2)

    # Additional warm tint toward aged yellow
    arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.08 + 18, 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] * 1.00 + 10, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] * 0.82,       0, 255)

    # Paper grain noise
    noise = np.random.normal(0, 6, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    result = Image.fromarray(arr)
    # Slight blur — aged maps lose fine detail
    return result.filter(ImageFilter.GaussianBlur(radius=0.6))


def to_faded(img: Image.Image) -> Image.Image:
    """
    Reduce saturation and contrast to simulate a sun-faded or water-damaged map.
    Composites on parchment background.
    """
    from PIL import ImageEnhance
    rgb = composite(img, BG_PARCHMENT)
    rgb = ImageEnhance.Color(rgb).enhance(0.35)       # desaturate
    rgb = ImageEnhance.Contrast(rgb).enhance(0.6)     # reduce contrast
    rgb = ImageEnhance.Brightness(rgb).enhance(1.15)  # slightly brighter (washed out)
    return rgb


# ---------------------------------------------------------------------------
# Spatial augmentations
# ---------------------------------------------------------------------------

def make_grid(
    images: list[Image.Image],
    rows: int,
    cols: int,
    bg: tuple[int, int, int] = BG_WHITE,
    padding: int = 4,
) -> Image.Image:
    """
    Tile a list of images into a (rows × cols) grid.

    Images are composited onto bg before tiling. They are all resized to the
    size of the first image. If fewer images than cells are provided, the list
    is cycled. Returns a single RGB image.

    Use this to create 'terrain region' patches: tiling multiple variants of
    the same hex class produces a composite that looks more like a map crop
    than a single isolated symbol.
    """
    if not images:
        raise ValueError("images list is empty")

    tile_w, tile_h = images[0].size
    cell_w = tile_w + padding
    cell_h = tile_h + padding
    canvas = Image.new("RGB", (cols * cell_w - padding, rows * cell_h - padding), bg)

    for idx in range(rows * cols):
        img = images[idx % len(images)]
        rgb = composite(img.resize((tile_w, tile_h), Image.LANCZOS), bg)
        row, col = divmod(idx, cols)
        canvas.paste(rgb, (col * cell_w, row * cell_h))

    return canvas
