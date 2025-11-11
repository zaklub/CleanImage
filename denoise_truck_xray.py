import argparse
from typing import Literal

import cv2
import numpy as np


MaskMode = Literal["otsu", "variance", "grabcut"]


def _compute_mask_otsu(
    img: np.ndarray, morph_kernel: int, dilate_iters: int
) -> np.ndarray:
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert if the brighter region got selected.
    if np.mean(img[mask == 255]) > np.mean(img[mask == 0]):
        mask = cv2.bitwise_not(mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=dilate_iters)
    return mask


def _compute_mask_variance(
    img: np.ndarray,
    morph_kernel: int,
    dilate_iters: int,
    variance_kernel: int,
    variance_percentile: float,
) -> np.ndarray:
    img_float = img.astype(np.float32)
    kernel_size = (variance_kernel, variance_kernel)
    mean = cv2.blur(img_float, kernel_size)
    mean_sq = cv2.blur(img_float * img_float, kernel_size)
    variance = np.clip(mean_sq - mean * mean, 0.0, None)

    threshold_value = np.percentile(variance, variance_percentile)
    mask = np.where(variance > threshold_value, 255, 0).astype(np.uint8)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_close)

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
    mask = cv2.dilate(mask, kernel_dilate, iterations=dilate_iters)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        mask[:] = 0
        cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, -1)

    return mask


def _compute_mask_grabcut(
    color_img: np.ndarray,
    morph_kernel: int,
    dilate_iters: int,
    vertical_margin_ratio: float,
    horizontal_margin_ratio: float,
    iterations: int,
) -> np.ndarray:
    h, w = color_img.shape[:2]
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)

    v_margin = int(max(1, vertical_margin_ratio * h))
    h_margin = int(max(1, horizontal_margin_ratio * w))

    mask[:v_margin, :] = cv2.GC_BGD
    mask[h - v_margin :, :] = cv2.GC_BGD
    mask[v_margin : h - v_margin, h_margin : w - h_margin] = cv2.GC_PR_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        color_img,
        mask,
        None,
        bgd_model,
        fgd_model,
        max(1, iterations),
        cv2.GC_INIT_WITH_MASK,
    )

    binary_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.dilate(binary_mask, kernel, iterations=dilate_iters)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        binary_mask[:] = 0
        cv2.drawContours(binary_mask, [max(contours, key=cv2.contourArea)], -1, 255, -1)

    return binary_mask


def denoise_truck_xray(
    input_path: str,
    mask_path: str = "mask_container.png",
    output_path: str = "xray_cleaned.png",
    inside_h: int = 8,
    outside_blur_kernel: int = 31,
    morph_kernel: int = 15,
    dilate_iters: int = 2,
    feather_blur_kernel: int = 31,
    feather_sigma: float = 15.0,
    mask_mode: MaskMode = "otsu",
    variance_kernel: int = 15,
    variance_percentile: float = 70.0,
    grabcut_iterations: int = 5,
    vertical_margin_ratio: float = 0.12,
    horizontal_margin_ratio: float = 0.05,
) -> None:
    """Auto-mask the truck/container region and selectively denoise the image."""
    color_img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if color_img is None:
        raise FileNotFoundError(f"Could not open input image: {input_path}")
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if mask_mode == "otsu":
        mask = _compute_mask_otsu(img, morph_kernel, dilate_iters)
    elif mask_mode == "variance":
        mask = _compute_mask_variance(
            img,
            morph_kernel=morph_kernel,
            dilate_iters=dilate_iters,
            variance_kernel=variance_kernel,
            variance_percentile=variance_percentile,
        )
    elif mask_mode == "grabcut":
        mask = _compute_mask_grabcut(
            color_img,
            morph_kernel=morph_kernel,
            dilate_iters=dilate_iters,
            vertical_margin_ratio=vertical_margin_ratio,
            horizontal_margin_ratio=horizontal_margin_ratio,
            iterations=grabcut_iterations,
        )
    else:
        raise ValueError(f"Unsupported mask_mode: {mask_mode}")

    inside = cv2.bitwise_and(img, img, mask=mask)
    outside_mask = cv2.bitwise_not(mask)
    outside = cv2.bitwise_and(img, img, mask=outside_mask)

    inside_denoised = cv2.fastNlMeansDenoising(inside, h=inside_h)

    blur_size = (outside_blur_kernel, outside_blur_kernel)
    outside_blur = cv2.GaussianBlur(outside, blur_size, 0)

    feather = (
        cv2.GaussianBlur(mask, (feather_blur_kernel, feather_blur_kernel), feather_sigma)
        / 255.0
    )
    result = (feather * inside_denoised + (1.0 - feather) * outside_blur).astype(np.uint8)

    cv2.imwrite(mask_path, mask)
    cv2.imwrite(output_path, result)
    print(f"Saved: {mask_path} and {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-mask the container region in a truck X-ray and selectively denoise it."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default="normalized_img1.png",
        help="Path to the input grayscale image (PNG, JPG, etc.).",
    )
    parser.add_argument(
        "--mask-path",
        default="mask_container.png",
        help="File path where the generated container mask will be written.",
    )
    parser.add_argument(
        "--output-path",
        default="xray_cleaned.png",
        help="File path where the cleaned X-ray will be written.",
    )
    parser.add_argument(
        "--inside-h",
        type=int,
        default=8,
        help="Strength for fastNlMeansDenoising inside the container region.",
    )
    parser.add_argument(
        "--outside-blur-kernel",
        type=int,
        default=31,
        help="Gaussian blur kernel size (must be odd) for the outside region.",
    )
    parser.add_argument(
        "--morph-kernel",
        type=int,
        default=15,
        help="Morphological kernel size (square) used to clean up the mask.",
    )
    parser.add_argument(
        "--dilate-iters",
        type=int,
        default=2,
        help="Number of dilation iterations to expand the mask.",
    )
    parser.add_argument(
        "--feather-blur-kernel",
        type=int,
        default=31,
        help="Gaussian blur kernel size (must be odd) for feathering the mask edges.",
    )
    parser.add_argument(
        "--feather-sigma",
        type=float,
        default=15.0,
        help="Gaussian sigma used for feathering the mask edges.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=["otsu", "variance", "grabcut"],
        default="otsu",
        help="Strategy used to build the container mask.",
    )
    parser.add_argument(
        "--variance-kernel",
        type=int,
        default=15,
        help="Kernel size (must be odd) for the local variance filter when mask-mode=variance.",
    )
    parser.add_argument(
        "--variance-percentile",
        type=float,
        default=70.0,
        help="Percentile threshold for variance masking (higher -> tighter mask).",
    )
    parser.add_argument(
        "--grabcut-iters",
        type=int,
        default=5,
        help="Number of GrabCut refinement iterations when mask-mode=grabcut.",
    )
    parser.add_argument(
        "--grabcut-vertical-margin",
        type=float,
        default=0.12,
        help="Vertical margin ratio (0-0.5) treated as definite background for GrabCut.",
    )
    parser.add_argument(
        "--grabcut-horizontal-margin",
        type=float,
        default=0.05,
        help="Horizontal margin ratio treated as probable foreground for GrabCut.",
    )

    args = parser.parse_args()
    denoise_truck_xray(
        input_path=args.input_path,
        mask_path=args.mask_path,
        output_path=args.output_path,
        inside_h=args.inside_h,
        outside_blur_kernel=args.outside_blur_kernel,
        morph_kernel=args.morph_kernel,
        dilate_iters=args.dilate_iters,
        feather_blur_kernel=args.feather_blur_kernel,
        feather_sigma=args.feather_sigma,
        mask_mode=args.mask_mode,  # type: ignore[arg-type]
        variance_kernel=args.variance_kernel,
        variance_percentile=args.variance_percentile,
        grabcut_iterations=args.grabcut_iters,
        vertical_margin_ratio=args.grabcut_vertical_margin,
        horizontal_margin_ratio=args.grabcut_horizontal_margin,
    )

