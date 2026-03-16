import numpy as np
import SimpleITK as sitk


def skull_strip(image):
    """
    Basic skull stripping using Otsu threshold + largest component extraction.

    image: numpy array (H, W, D)
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be numpy array")

    sitk_image = sitk.GetImageFromArray(image)

    # Otsu thresholding
    mask = sitk.OtsuThreshold(
        sitk_image,
        0,
        1,
        200
    )

    # Keep largest connected component
    cc = sitk.ConnectedComponent(mask)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)

    largest_label = max(
        stats.GetLabels(),
        key=lambda l: stats.GetPhysicalSize(l)
    )

    brain_mask = sitk.BinaryThreshold(
        cc,
        lowerThreshold=largest_label,
        upperThreshold=largest_label,
        insideValue=1,
        outsideValue=0
    )

    stripped = sitk_image * brain_mask

    return sitk.GetArrayFromImage(stripped)