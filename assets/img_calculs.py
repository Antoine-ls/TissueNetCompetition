"""
This py file contains functions that preprocess the image, they are:
- segmentation functions
- denoise functions
- contours functions

"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import collections


def average_color(img: np.array, use_torch=False):
    if use_torch:
        img = torch.Tensor(img)
        res = torch.sum(img) / np.prod(img.shape())
    else:
        res = np.average(img)

    return res


def seg_kmeans(a_img_in: np.array, disp=False, denoise_f=None):
    """
    This function segment the image by kmeans
    :param a_img_in: input rgb image(np.array)
    :param disp: flag, whether to plot the result
    :param cvt_rgb: flag, whether to to BGR2RGB convert
        :param denoise_f: denoise functions
    :return: a generated mask(np.array) by kmeans method
    """

    img_flat = a_img_in.reshape((a_img_in.shape[0] * a_img_in.shape[1], 3)).astype(np.float32)

    # iteration parameters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 10, 1)
    flags = cv2.KMEANS_PP_CENTERS

    # Do kmeans
    compactness, labels, centers = cv2.kmeans(img_flat, 2, None, criteria, 10, flags)

    # Reshape
    img_out = labels.reshape((a_img_in.shape[0], a_img_in.shape[1])).astype(np.uint8)

    if denoise_f is not None:
        if isinstance(denoise_f, collections.Iterable):
            for f in denoise_f:
                img_out = f(img_out)
        else:
            img_out = denoise_f(img_out)

    if disp:
        plt.subplot(121), plt.imshow(a_img_in), plt.title('input')
        plt.subplot(122), plt.imshow(img_out, 'gray'), plt.title('kmeans')
        plt.show()

    return img_out


def seg_canny(a_img_in: np.array, disp=False, denoise_f=None):
    """
    This function implement segmentation by cv2.canny()
    :param a_img_in: input rgb image(np.array)
    :param disp: flag, whether to plot the result
    :param cvt_rgb: flag, whether to to BGR2RGB convert
    :param denoise_f: denoise functions
    :return: a generated mask(np.array) by kmeans method
    """

    # canny
    img_out = cv2.Canny(a_img_in, 125, 256)

    if disp:
        plt.subplot(121), plt.imshow(a_img_in), plt.title('input')
        plt.subplot(122), plt.imshow(img_out, 'gray'), plt.title('kmeans')
        plt.show()

    return img_out


def seg_gray_threshold(a_img_in: np.array, disp=False, denoise_f=None):
    """
    :param a_img_in: input rgb image(np.array)
    :param disp: flag, whether to plot the result
    :param cvt_rgb: flag, whether to to BGR2RGB convert
    :param denoise_f: denoise functions
    :return: a generated mask(np.array) by threshold method
    """
    # convertion

    img_in = cv2.cvtColor(a_img_in, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    # Perform a gaussion filter
    img_blur = cv2.GaussianBlur(img_in, (5, 5), 0)

    # use cv2.threshold to generate a mask
    re3, img_out = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if denoise_f is not None:
        if isinstance(denoise_f, collections.Iterable):
            for f in denoise_f:
                img_out = f(img_out)
        else:
            img_out = denoise_f(img_out)

    if disp:
        plt.subplot(121), plt.imshow(a_img_in), plt.title('input')
        plt.subplot(122), plt.imshow(img_out, 'gray'), plt.title('threshhold')
        plt.show()

    return img_out


def seg_hsv_threshold(a_img_in: np.array, disp=False, denoise_f=None):
    """
    :param a_img_in: input rgb image(np.array)
    :param disp: flag, whether to plot the result
    :param cvt_rgb: flag, whether to to BGR2RGB convert
    :param denoise_f: denoise functions
    :return: a generated mask(np.array) by threshold method
    """
    # convertion

    img_in = cv2.cvtColor(a_img_in, cv2.COLOR_BGR2HSV).astype(np.uint8)
    # Perform a gaussion filter
    purple_lower = (78, 20, 20)
    purple_upper = (165, 255, 255)

    # use cv2.threshold to generate a mask
    img_out = cv2.inRange(img_in, purple_lower, purple_upper)

    kernel = np.ones((5, 5), np.uint8)
    img_out = cv2.dilate(img_out, kernel, iterations=1)

    if denoise_f is not None:
        if isinstance(denoise_f, collections.Iterable):
            for f in denoise_f:
                img_out = f(img_out)
        else:
            img_out = denoise_f(img_out)

    if disp:
        plt.subplot(121), plt.imshow(a_img_in), plt.title('input')
        plt.subplot(122), plt.imshow(img_out, 'gray'), plt.title('threshhold')
        plt.show()

    return img_out


def denoise_bilatera(a_img_in: np.array, disp=False):
    """
    This function denoise a mask by bilateralFilter
    :param a_img_in: input binary image(np.array)
    :param disp: flag, whether to plot the result
    :return: a denoised mask(np.array) by threshold method
    """
    # apply filters
    img_blur = cv2.bilateralFilter(a_img_in, 9, 80, 80)
    fill = img_blur.copy()
    h, w = img_blur.shape[: 2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(fill, mask, (90, 90), 255)
    fill_INV = cv2.bitwise_not(fill)
    img_out = img_blur | fill_INV

    if disp:
        plt.subplot(121), plt.imshow(a_img_in, 'gray'), plt.title('input')
        plt.subplot(122), plt.imshow(img_out, 'gray'), plt.title('denoised_bilatera')
        plt.show()

    return img_out


def denoise_baweraopen(a_img_in: np.array, disp=False, a_size=10):
    """
    This function denoise a mask by baweraopen
    :param a_img_in:input binary image(np.array)
    :param a_size: size of area desired to remove
    """
    img_out = a_img_in.copy()
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(a_img_in)
    for i in range(1, nlabels - 1):
        regions_size = stats[i, 4]
        if regions_size < a_size:
            x0 = stats[i, 0]
            y0 = stats[i, 1]
            x1 = stats[i, 0] + stats[i, 2]
            y1 = stats[i, 1] + stats[i, 3]
            for row in range(y0, y1):
                for col in range(x0, x1):
                    if labels[row, col] == i:
                        img_out[row, col] = 0
    if disp:
        plt.subplot(121), plt.imshow(a_img_in), plt.title('input')
        plt.subplot(122), plt.imshow(img_out, 'gray'), plt.title('denoised_baweraopen')
        plt.show()

    return img_out


def denoise_erode(a_img_in: np.array, disp=False):
    """
    This function denoise a mask by erosion
    :param a_img_in: input binary image(np.array)
    :param disp: flag, whether to plot the result
    :return: a denoised mask(np.array) by threshold method
    """
    # apply filters
    kernel = np.ones((3, 3), np.uint8)
    # iteration的值越高，模糊程度(腐蚀程度)就越高 呈正相关关系且只能是整数
    img_out = cv2.erode(a_img_in, kernel, iterations=2)

    if disp:
        plt.subplot(121), plt.imshow(a_img_in, 'gray'), plt.title('input')
        plt.subplot(122), plt.imshow(img_out, 'gray'), plt.title('denoised_bilatera')
        plt.show()

    return img_out


def apply_mask(a_img_in: np.array, a_mask_in: np.array, disp=False):
    """
    This function applies a mask to the input image
    :param a_img_in: input rgb/grey image(np.array)
    :param a_mask_in: mask binary image(np.array), reserved area represented by 1
    :param disp: flag, whether to plot the result
    :return: result
    """
    # if the input image is rgb, expand the dimension of mask
    if len(a_img_in.shape) == 3:
        mask_3d = np.empty_like(a_img_in)
        for i in range(3):
            mask_3d[:, :, i] = a_img_in
        img_out = np.dot(a_img_in, mask_3d)
    else:
        img_out = np.dot(a_img_in, a_mask_in)

    if disp:
        plt.subplot(121), plt.imshow(a_img_in), plt.title('input')
        plt.subplot(122), plt.imshow(img_out, 'gray'), plt.title('bounding_box')
        plt.show()

    return img_out


def get_boxes_contours(a_img_in: np.array, a_mask_in: np.array, disp=False, draw_contours=False):
    """
    This function generates the bounding boxes
    :param a_img_in: input rgb image (np.array)
    :param a_mask_in: input binary image (np.array)
    :param disp: flag, whether to plot the result
    :return: boxes[(x, y, w, h),...] list of tuples, contours [np.array(n,1,2),...], hierarchy
    """
    # detect contours and boxes
    contours, hierarchy = cv2.findContours(a_mask_in, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for i, contour in enumerate(contours):
        boxes.append(cv2.boundingRect(contour))

    if disp:
        img_out = a_img_in.copy().astype(np.uint8)
        if draw_contours:
            # Draw contours
            cv2.drawContours(img_out, contours, -1, (0, 255, 0), 5)
        for box in boxes:
            x, y, w, h = box
            cv2.rectangle(img_out, (x, y), (x + w, y + h), (255, 0, 0), 5)
        plt.subplot(121), plt.imshow(a_img_in), plt.title('input')
        plt.subplot(122), plt.imshow(img_out), plt.title('bounding_box')
        plt.show()

    return boxes, contours, hierarchy


def get_polygons_contours(a_img_in: np.array, a_mask_in: np.array, disp=False):
    """
    This function generates the bounding polygens
    :param a_img_in: input rgb image (np.array)
    :param a_mask_in: input binary image (np.array)
    :param disp: flag, whether to plot the result

    :return:
    """
    contours, hierarchy = cv2.findContours(a_mask_in, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygens = [cv2.convexHull(contour) for contour in contours]

    if disp:
        img_out = a_img_in.copy().astype(np.uint8)
        cv2.polylines(img_out, polygens, True, (0, 0, 255), 2)

        plt.subplot(121), plt.imshow(a_img_in), plt.title('input')
        plt.subplot(122), plt.imshow(img_out), plt.title('bounding_polygen')
        plt.show()

    return polygens, contours, hierarchy


def reduce_boxes(boxes, size):
    boxes_reduced = [box for box in boxes if box[3] > size or box[4] > size]
    return boxes_reduced
