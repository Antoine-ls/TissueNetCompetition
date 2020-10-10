"""
This py script generate a [W, H, D] array representing the tif/jpg image.
"""
import pyvips
import numpy as np
import cv2


def np_from_tif(a_path: str, a_page=4, a_crop: tuple = None):
    """
    :param a_path: path of tif file
    :param a_page: page of tif file
    :param a_crop: (x, y, region_width, region_height)
    :return:
    """
    slide = pyvips.Image.new_from_file(a_path, page=a_page)
    if a_crop is not None:
        region = pyvips.Region.new(slide).fetch(*a_crop)
        array_out = np.ndarray(
            buffer=region.write_to_memory(),
            dtype=np.uint8,
            shape=(region.height, region.width, region.bands)
        )
        return array_out

    else:
        array_out = np.ndarray(
            buffer=slide.write_to_memory(),
            dtype=np.uint8,
            shape=(slide.height, slide.width, slide.bands)
        )
        return array_out


def np_from_jpg(a_path: str, a_cvt_color=False):
    """
    :param a_path: path to jpg file
    :param a_cvt_color: flag, whether to convert color format
    :return:
    """
    array_out = cv2.imread(a_path, cv2.IMREAD_COLOR)
    if a_cvt_color:
        array_out = cv2.cvtColor(array_out, cv2.COLOR_BGR2RGB)

    return array_out
