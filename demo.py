from matplotlib import pyplot as plt
from img_calculs import seg_threshold, seg_kmeans, seg_canny, get_boxes_contours, get_polygens_contours
from img_calculs import denoise_bilatera, denoise_baweraopen, denoise_erode
from img2np import np_from_tif, np_from_jpg
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if __name__ == '__main__':
    img = np_from_tif('../data/tif_images/C13_B156_S11.tif', 7)

    # img = np_from_jpg('../data/downsampled_images/C01_B202_S01.jpg', True)
    mask = seg_threshold(img, disp=False, cvt_rgb=False, denoise_f=(denoise_erode, denoise_bilatera))
    boxes, contours, _ = get_boxes_contours(img, mask, disp=True)
    # hulls, contours, _ = get_polygens_contours(img, mask, disp=True)



    print('Finished')

    # 等待修改：
    # 筛掉过小的ROI区域？
    # 生成多边形的ROI区域，然后采用更合适的切片算法

