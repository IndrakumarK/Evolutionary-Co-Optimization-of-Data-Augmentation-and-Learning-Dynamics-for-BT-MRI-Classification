import SimpleITK as sitk
import numpy as np


def resample(image, new_spacing=(1.0, 1.0, 1.0)):
    """
    Resample MRI volume to isotropic spacing.

    image: numpy array (H, W, D)
    new_spacing: desired voxel spacing
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be numpy array")

    sitk_image = sitk.GetImageFromArray(image)

    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()

    new_size = [
        int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetOutputSpacing(new_spacing)
    resample_filter.SetSize(new_size)
    resample_filter.SetInterpolator(sitk.sitkLinear)
    resample_filter.SetOutputDirection(sitk_image.GetDirection())
    resample_filter.SetOutputOrigin(sitk_image.GetOrigin())

    resampled = resample_filter.Execute(sitk_image)

    return sitk.GetArrayFromImage(resampled)