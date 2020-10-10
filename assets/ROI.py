"""
This file contains a coordinate convertor, it has 3 (useful) methods


"""
import numpy as np
import matplotlib.pyplot as plt


class ROI:
    """
    _boxes stores box coordinates
    """

    def __init__(self):
        """
        The coordinates are stored in a format where $(x,y) \in [0,1]^2$, left-up corner->(0,0)
        :return:
        """
        self._boxes_normalized_lu = dict()
        self._boxes_normalized = dict()

    def set_boxes(self, name: str, W: float, H: float, boxes: list, coordinate='lu'):
        """
        set boxes
        :param name: name of image
        :param W: width of the image
        :param H: height of the image
        :param boxes: [(x, y, w, h)...] list of tuple
        :param coordinate: coordinate of the input "lu" for left-up, "lb" for left-bottom
        :return: None
        """
        boxes_np = np.array(boxes).astype(np.float64)
        if coordinate == 'lu':
            boxes_np[:, [0, 2]] = boxes_np[:, [0, 2]] / W
            boxes_np[:, 1] = H - boxes_np[:, 1] - boxes_np[:, 3]
            boxes_np[:, [1, 3]] = boxes_np[:, [1, 3]] / H
            self._boxes_normalized[name] = boxes_np

        elif coordinate == 'lb':
            boxes_np[:, [0, 2]] = boxes_np[:, [0, 2]] / W
            boxes_np[:, [1, 3]] = boxes_np[:, [1, 3]] / H
            self._boxes_normalized[name] = boxes_np

        else:
            raise NotImplementedError

    def get_boxes_normalized(self, name, coordinate='lb'):
        """
        :param name: name of the image that you want to know roi
        :param coordinate: coordinate of the input "lu" for left-up, "lb" for left-bottom
        :return: normailzed coordinate [[x, y, w, h]] in an np.array
        """
        if name not in self._boxes_normalized.keys():
            return np.zeros(shape=(1, 4))
        else:
            res = np.array(self._boxes_normalized[name])
            if coordinate == 'lb':
                return res
            elif coordinate == 'lu':
                res[:, 1] = 1 - res[:, 1] - res[:, 3]
                return res
            else:
                raise NotImplementedError

    def plot(self, name):
        """
        Visualization of
        :param name:
        :return:
        """
        b = self.get_boxes_normalized(name, 'lb')
        x = b[:, 0] + b[:, 2] / 2
        y = b[:, 1] + b[:, 3] / 2
        plt.scatter(x, y, s=1)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.show()


if __name__ == '__main__':
    from img_calculs import seg_kmeans, seg_canny, get_boxes_contours
    from img_calculs import denoise_bilatera, denoise_erode
    from img2np import np_from_tif, np_from_jpg

    img = np_from_jpg('../data/downsampled_images/C06_B001_S21.jpg')
    # img = np_from_tif('../data/tif_images/C01_B103_S01.tif', 7)
    mask = seg_kmeans(img, disp=False, denoise_f=(denoise_erode, denoise_bilatera))
    boxes, contours, _ = get_boxes_contours(img, mask, disp=True)
    R = ROI()
    R.set_boxes(name='C13_B156_S11.tif', W=img.shape[1], H=img.shape[0], boxes=boxes, coordinate='lu')
    R.plot('C13_B156_S11.tif')
    B = R.get_boxes_normalized('C13_B156_S11.tif', 'lu')
    print('Finished')
