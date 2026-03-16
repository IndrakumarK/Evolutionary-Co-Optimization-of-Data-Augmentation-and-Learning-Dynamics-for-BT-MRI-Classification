import SimpleITK as sitk
import numpy as np


def reorient(image):
    """
    Reorient MRI volume to canonical RAS orientation.

    image: numpy array (H, W, D)
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be numpy array")

    sitk_image = sitk.GetImageFromArray(image)

    # Reorient to RAS
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation("RAS")

    reoriented = orient_filter.Execute(sitk_image)

    return sitk.GetArrayFromImage(reoriented)