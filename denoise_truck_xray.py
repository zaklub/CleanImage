import cv2
import numpy as np


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
) -> None:
    """Auto-mask the truck/container region and selectively denoise the image."""
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not open input image: {input_path}")
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Step 2: auto-mask the container via Otsu threshold + morphology
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert if the brighter region got selected
    if np.mean(img[mask == 255]) > np.mean(img[mask == 0]):
        mask = cv2.bitwise_not(mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=dilate_iters)

    # Step 3: selective denoise
    inside = cv2.bitwise_and(img, img, mask=mask)
    outside_mask = cv2.bitwise_not(mask)
    outside = cv2.bitwise_and(img, img, mask=outside_mask)

    inside_denoised = cv2.fastNlMeansDenoising(inside, h=inside_h)

    blur_size = (outside_blur_kernel, outside_blur_kernel)
    outside_blur = cv2.GaussianBlur(outside, blur_size, 0)

    # Step 4: feathered blend
    feather = (
        cv2.GaussianBlur(mask, (feather_blur_kernel, feather_blur_kernel), feather_sigma)
        / 255.0
    )
    result = (feather * inside_denoised + (1.0 - feather) * outside_blur).astype(np.uint8)

    # Step 5: save outputs
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(output_path, result)
    print(f"Saved: {mask_path} and {output_path}")


if __name__ == "__main__":
    denoise_truck_xray("normalized_img1.png")

