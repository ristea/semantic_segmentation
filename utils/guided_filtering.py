import numpy as np
import time
import cv2
from cv2.ximgproc import guidedFilter


def fillHole(mask, radius, n):
    kernel = np.ones((radius, radius), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=n)


def removeIland(mask, radius, n):
    kernel = np.ones((radius, radius), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=n)


def doMorphSmoothing(mask, radius, n, maxDim):
    oriShape = mask.shape[::-1]
    if max(oriShape) > maxDim:
        resizeShape = tuple([dim * maxDim // max(oriShape) for dim in oriShape])
        maskResize = cv2.resize(mask, resizeShape, interpolation=cv2.INTER_AREA)
    else:
        maskResize = mask
    maskResize = fillHole(maskResize, radius, n)
    maskResize = removeIland(maskResize, radius, n)
    if max(oriShape) > maxDim:
        maskResize = cv2.resize(maskResize, oriShape, interpolation=cv2.INTER_LINEAR)
    return maskResize


def doGuidedFiltering(img, mask, iters=3, maxDim=400):
    r = 10
    eps = 1e-4
    eps *= 255 * 255

    assert (img.shape[:2] == mask.shape)
    oriShape = mask.shape[::-1]
    maxVal = mask.max()
    if maxVal > 0:
        mask = (mask.astype(np.int32) * 255 // maxVal).astype(np.uint8)

    if max(oriShape) > maxDim:
        resizeShape = tuple([dim * maxDim // max(oriShape) for dim in oriShape])
    else:
        resizeShape = oriShape
    # resizeShape = tuple([dim // s * s for dim in resizeShape])
    imgResize = cv2.resize(img, resizeShape, interpolation=cv2.INTER_AREA)
    maskResize = cv2.resize(mask, resizeShape, interpolation=cv2.INTER_LINEAR)

    imgResize = cv2.GaussianBlur(imgResize, (3, 3), 0.5)
    ret, maskResize = cv2.threshold(maskResize, 127, 255, cv2.THRESH_BINARY)

    for i in range(iters):
        maskGF = guidedFilter(imgResize, maskResize, r, eps)
        ret, maskResize = cv2.threshold(maskGF, 127, 255, cv2.THRESH_BINARY);

    maskGF = cv2.resize(maskGF, oriShape, interpolation=cv2.INTER_CUBIC)
    ret, maskGF = cv2.threshold(maskGF, 255, 255, cv2.THRESH_TRUNC)
    ret, maskGF = cv2.threshold(maskGF, 0, 255, cv2.THRESH_TOZERO)

    #region = (mask > 0.9 * 255) | (mask < 0.1 * 255)
    #maskGF[region] = mask[region]

    maskGF = doMorphSmoothing(maskGF, 1, 20, maxDim)
    ret, maskGF = cv2.threshold(maskGF, 127, 255, cv2.THRESH_BINARY)
    return maskGF


if __name__ == "__main__":
    mask = cv2.imread('output.png', cv2.IMREAD_GRAYSCALE)
    image = cv2.imread('images/girl.jpg', cv2.IMREAD_COLOR)
    start = time.time()
    mask_smoothed = doGuidedFiltering(image, mask)
    end = time.time()
    print(end - start)
    cv2.imwrite('mask_smoothed.png', mask_smoothed)

    cv2.imshow('', mask_smoothed)
    cv2.waitKey(0)

    image[mask_smoothed == 0] = 0
    cv2.imwrite('layerized_smoothed.jpg', image)

