#!/usr/bin/env python
import sys
import copy
import math

import cv2
import matplotlib.pylab as plt


def plot(img, threshold, blur, contour_img, bbox_img):
    # plot
    plt.subplot(511)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(512)
    plt.title('Threshold Image')
    plt.imshow(threshold, cmap='gray')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(513)
    plt.title('Blur Image')
    plt.imshow(blur, cmap='gray')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(514)
    plt.title('Contour Image')
    plt.imshow(contour_img, cmap='gray')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(515)
    plt.title('Bounding Box Image')
    plt.imshow(bbox_img, cmap='gray')
    plt.xticks([])
    plt.yticks([])

    plt.show()


def save_bbox(image_index, bbox_img, top_left, bottom_right):
    # save bounding box
    x = top_left[0] if top_left[0] > 0 else 0
    y = top_left[1] if top_left[1] > 0 else 0
    w = math.fabs(bottom_right[0] - top_left[0])
    h = math.fabs(bottom_right[1] - top_left[1])

    side = max(w, h)

    bbox_img = img[y:y + side, x:x + side]
    image_path = "/data/{0}.jpg".format(image_index)
    cv2.imwrite(image_path, bbox_img)


if __name__ == "__main__":
    start = int(sys.argv[1])
    end = int(sys.argv[2])

    for image_index in range(start, end + 1):
        img_file = "/data/imgs/w_{0}.jpg".format(image_index)
        print "processing {0}".format(img_file)

        # load image
        img = cv2.imread(img_file)

        # threshold
        gray = cv2.cvtColor(copy.deepcopy(img), cv2.COLOR_BGR2GRAY)
        ret, threshold = cv2.threshold(
            gray,
            200,
            255,
            cv2.THRESH_BINARY
        )

        # blur
        blur = cv2.blur(copy.deepcopy(threshold), (20, 20))

        # contour
        contour_img = copy.deepcopy(img)
        contours, hierarchy = cv2.findContours(
            copy.deepcopy(blur),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # obtain contour areas
        contour_areas = []
        for contour in contours:
            contour_areas.append(cv2.contourArea(contour))

        best_contours = []
        sorted_contour_areas = copy.deepcopy(contour_areas)
        sorted_contour_areas.sort(reverse=True)
        for i in range(2):
            index = contour_areas.index(sorted_contour_areas[i])
            c = contours[index]
            best_contours.append(c)

        # draw contours
        cv2.drawContours(contour_img, best_contours, -1, (255, 0, 0), 20)

        # obtain bounding box
        c1 = best_contours[0]
        c2 = best_contours[1]
        c1x, c1y, c1w, c1h = cv2.boundingRect(c1)
        c2x, c2y, c2w, c2h = cv2.boundingRect(c2)

        top_left = [min(c1x, c2x), min(c1y, c2y)]
        bottom_right = [
            max(c1x + c1w, c2x + c2w),
            max(c1y + c1h, c2y + c2h)
        ]
        padding = 0
        top_left[0] -= padding
        top_left[1] -= padding
        bottom_right[0] += padding
        bottom_right[1] += padding
        bbox_img = copy.deepcopy(img)
        cv2.rectangle(
            bbox_img,
            tuple(top_left),
            tuple(bottom_right),
            (255, 0, 0),
            20
        )

        plot(img, threshold, blur, contour_img, bbox_img)
        # save_bbox(image_index, bbox_img, top_left, bottom_right)
