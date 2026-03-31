from typing import Tuple

import numpy
import pandas


def check_for_xy_squareness(bbox: tuple[int, int, int, int]) -> float:
    """
    This function returns the ratio of the x length to the y length
    A value of 1 indicates a square bbox is present

    Parameters
    ----------
    bbox : The bbox to check
        (y_min, x_min, y_max, x_max)
        Where each value is an int representing the pixel coordinate of the bbox in that dimension

    Returns
    -------
    float
        The ratio of the y length to the x length of the bbox. A value of 1 indicates a square bbox.
    """
    y_min, x_min, y_max, x_max = bbox
    x_length = x_max - x_min
    if x_length == 0:
        raise ValueError(
            "Cannot compute xy squareness for bbox with zero width in x dimension "
            f"(bbox={bbox})."
        )
    xy_squareness = (y_max - y_min) / x_length
    return xy_squareness


def square_off_xy_crop_bbox(
    bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """
    Adjust the bbox to be square in the XY plane.

    The function computes the new bbox from the current X/Y dimensions.

    Parameters
    ----------
    bbox : tuple[int, int, int, int]
        The bbox to adjust:
        (y_min, x_min, y_max, x_max)

        Each value is an integer pixel coordinate in that dimension.

    Returns
    -------
    tuple[int, int, int, int]
        The adjusted bbox that is square in the XY plane:
        (new_y_min, new_x_min, new_y_max, new_x_max)

        Each value is an integer pixel coordinate in that dimension.
    """
    ymin, xmin, ymax, xmax = bbox
    # first find the larger dimension between x and y
    x_size = xmax - xmin
    y_size = ymax - ymin
    if x_size > y_size:
        # need to expand y dimension
        new_ymin = int(ymin - (x_size - y_size) / 2)
        new_ymax = int(ymax + (x_size - y_size) / 2)
        return (new_ymin, xmin, new_ymax, xmax)
    elif y_size > x_size:
        # need to expand x dimension
        new_xmin = int(xmin - (y_size - x_size) / 2)
        new_xmax = int(xmax + (y_size - x_size) / 2)
        return (ymin, new_xmin, ymax, new_xmax)
    else:
        # already square
        return bbox


def extract_x_y_centroid_from_image_based_profile(
    df: pandas.DataFrame,
    label: int,
    label_column_name: str = "Metadata_Nuclei_Number_Object_Number",
    centroid_column_names: Tuple[str, str] = (
        "Metadata_Cells_AreaShape_Center_Y",
        "Metadata_Cells_AreaShape_Center_X",
    ),
) -> Tuple[float, float]:

    """
    This function extracts the bbox from the image-based profile df

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the image-based profile for a single cell, which includes the bounding box coordinates for the cytoplasm.
    label : int
        The label of the object for which to extract the bounding box.

    Returns
    -------
    tuple[float, float, float, float]
        The bounding box coordinates (xmin, ymin, xmax, ymax) for the cytoplasm.
    """
    x_centroid = df.loc[df[label_column_name] == label, centroid_column_names[1]].item()
    y_centroid = df.loc[df[label_column_name] == label, centroid_column_names[0]].item()
    return x_centroid, y_centroid


def change_bbox_dtype_to_integer(
    bbox: tuple[float, float, float, float]
) -> tuple[int, int, int, int]:
    """
    This function changes the data type of the bbox coordinates to integers

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        The bounding box coordinates (xmin, ymin, xmax, ymax) for the cytoplasm.

    Returns
    -------
    tuple[int, int, int, int]
        The bounding box coordinates (xmin, ymin, xmax, ymax) for the cytoplasm with integer data type.
    """
    return tuple(map(int, bbox))


def crop_from_centroid(
    center_x: float, center_y: float, image_shape: tuple[int, int], radius=30
) -> tuple[int, int, int, int]:
    """
    Generate a crop_bbox from a center x,y and radius from that point

    Parameters
    ----------
    center_x : float
        The center x coordinate of the crop box
    center_y : float
        The center y coordinate of the crop box
    image_shape : tuple[int, int]
        The shape of the image
    radius : int, optional
        The radius of the crop box, by default 30

    Returns
    -------
    tuple[int, int, int, int]
        The crop bbox in the format (ymin, xmin, ymax, xmax)
    """
    # contruct a bbox
    radius = 30
    bbox = (
        center_y - radius,
        center_x - radius,
        center_y + radius,
        center_x + radius,
    )
    # if the box is outside of the image bounds then shift the box but keep it a box
    # if not check_for_xy_squareness(bbox):

    # find the shift needed to move the box inside the image bounds
    y_shift = 0
    x_shift = 0
    if bbox[0] < 0:
        y_shift = -bbox[0]
    if bbox[1] < 0:
        x_shift = -bbox[1]
    if bbox[2] > image_shape[0]:
        y_shift = image_shape[0] - bbox[2]
    if bbox[3] > image_shape[1]:
        x_shift = image_shape[1] - bbox[3]
    # apply the shift to the box
    bbox = (
        bbox[0] + y_shift,
        bbox[1] + x_shift,
        bbox[2] + y_shift,
        bbox[3] + x_shift,
    )
    # check if the box is now a square
    if not check_for_xy_squareness(bbox):
        print(f"Box is not square after shifting: {bbox}")

    return change_bbox_dtype_to_integer(bbox)
