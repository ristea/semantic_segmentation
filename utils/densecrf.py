import pydensecrf.densecrf as dcrf
from PIL import Image
import numpy as np
import cv2
# import time

def resize_short(img, dim):
    w, h = img.size
    if w > h:
        wr = dim
        hr = int(h / w * dim)
    else:
        wr = int(w / h * dim)
        hr = dim
    return img.resize((wr, hr), resample=Image.BILINEAR)

def histDist(img, maskFg, maskBg, channels):
    histFg = cv2.calcHist([img], channels, maskFg, [6], [0, 255], accumulate=False)
    histBg = cv2.calcHist([img], channels, maskBg, [6], [0, 255], accumulate=False)
    cv2.normalize(histFg, histFg, 1, 0, cv2.NORM_L1)
    cv2.normalize(histBg, histBg, 1, 0, cv2.NORM_L1)
    dist = cv2.compareHist(histFg, histBg, cv2.HISTCMP_BHATTACHARYYA)
    return dist

def color_hist_adaptive(img, mask):
    img = img[:, :, ::-1]
    _, maskFg = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    maskBg = 255 - maskFg
    distB = histDist(img, maskFg, maskBg, [0])
    distG = histDist(img, maskFg, maskBg, [1])
    distA = histDist(img, maskFg, maskBg, [2])
    distAll = (distB + distG + distA)/3
    bg_thresh = distAll * (-137.3179) + 85.4519
    if distAll < 0.5:
        fg_thresh = distAll * 644.0947 - 148.3629
    else:
        fg_thresh = distAll * 95.0949 + 122.1442
    bg_thresh = max(5.0, min(64.0, bg_thresh))
    fg_thresh = max(137.0, min(229.0, fg_thresh))
    return bg_thresh, fg_thresh

def unary_from_labels(labels, n_labels, gt_prob=0.5):
    labels = labels.flatten()
    u_energy = -np.log( 1.0 / n_labels )
    n_energy = -np.log((1.0 - gt_prob) / (n_labels - 1))
    p_energy = -np.log(gt_prob)
    U = np.full((n_labels, len(labels)), n_energy, dtype=np.float32)
    U[labels, np.arange(U.shape[1])] = p_energy
    U[:, labels == -1] = u_energy
    return U

def densecrf(img, mask, dim=512):
    if dim != None:
        img = resize_short(img, dim)
    w, h = img.size
    mask = mask.resize((w, h), resample=Image.BILINEAR)
    img_arr = np.array(img, dtype=np.uint8)
    mask_arr = np.array(mask, dtype=np.uint8)
    # start = time.time()
    
    bg_thresh, fg_thresh = color_hist_adaptive(img_arr, mask_arr)
    
    label_map = -np.ones(mask_arr.shape, dtype=np.int32)
    label_map[mask_arr < bg_thresh] = 0
    label_map[mask_arr > fg_thresh] = 1

    rst = np.zeros(mask_arr.shape, dtype=np.uint8)+128
    rst[mask_arr < bg_thresh] = 0
    rst[mask_arr > fg_thresh] = 255

    if 128 not in rst:
        return Image.fromarray(rst)

    unary = unary_from_labels(label_map, 3)
    d = dcrf.DenseCRF2D(w, h, 3)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    # change compat to 5 to make is sharper
    d.addPairwiseBilateral(sxy=max(w, h)//4+20, srgb=13, rgbim=img_arr, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(5)
    out = np.argmax(Q, axis=0).reshape(mask_arr.shape).astype(np.uint8)
    for i in range(h):
        for j in range(w):
            if rst[i][j] == 128:
                if out[i][j] == 0:
                    rst[i][j] = 0
                else:
                    rst[i][j] = 255
    # print(time.time() - start)
    rst = remove_iland(rst, 3)
    return Image.fromarray(rst)

def fill_hole(mask, radius):
    kernel = np.ones((radius,radius),np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def remove_iland(mask, radius):
    kernel = np.ones((radius,radius),np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

if __name__ == '__main__':
    img = Image.open("img.jpg")
    mask = Image.open("mask.png")
    new_mask = densecrf(img, mask)
    new_mask.save('out.png')