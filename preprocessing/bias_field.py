import SimpleITK as sitk
import numpy as np


def bias_correction(image):
    """
    Apply N4 bias-field correction using SimpleITK.

    image: numpy array (H, W) or (H, W, D)
    """

    if isinstance(image, np.ndarray):
        sitk_image = sitk.GetImageFromArray(image)
    else:
        raise TypeError("Input must be numpy array")

    mask_image = sitk.OtsuThreshold(
        sitk_image,
        0,
        1,
        200
    )

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = corrector.Execute(sitk_image, mask_image)

    corrected_array = sitk.GetArrayFromImage(corrected)

    return corrected_array