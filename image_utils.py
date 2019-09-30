import json
import shutil
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


def read_image(path, color_scheme=0):
    """
    Function to read an image at a given path
    :param path: - path to the image - string
    :param color_scheme: color scheme of the image to be read - int (default is 0 - grayScale)
    :return: name, img - name of the image and the image itself
    """
    img = cv2.imread(path, color_scheme)
    return path.split('\\')[-1], img


def get_files(img_dir):
    """
    Function to get all files from a directory
    :param img_dir: directory where the files are stored
    :return: three lists containing, images(jpg, jpeg, gif, png, pgm), mask_files(.bmp), xmls(.xml, .gt, .txt)
    """
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls


def list_files(in_path):
    """
    Function to get all files from a directory
    :param img_dir: directory where the files are stored
    :return: three lists containing, paths to all -> images(jpg, jpeg, gif, png, pgm), mask_files(.bmp), xmls(.xml, .gt, .txt)
    """
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    return img_files, mask_files, gt_files


def extract_chars(img):
    """
    Function to extract characters from an image
    :param img: input image - np array
    :return: a list containig all characters from the image
    """

    bw_image = img
    bw_image = cv2.Canny(bw_image, 100, 200)
    contours = cv2.findContours(bw_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    coordinates = []
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x, y, w, h = x - 2, y - 2, w + 4, h + 4
        coordinates.append((x, y, w, h))
        box = img[y:y + h, x:x + w]
        kernel = np.ones((3, 3), np.uint8)
        if box.shape[0] > 0 and box.shape[1] > 0:
            box = cv2.morphologyEx(box, cv2.MORPH_CLOSE, kernel)
            box = cv2.bitwise_not(box)
        bounding_boxes.append(box)

    return bounding_boxes, coordinates


def process_image(img):
    """
    Function to process the image before character extraction - adaptive binarization
    :param img: input image - np array
    :return: the processed image - binary image
    """
    if len(img.shape) != 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    ret, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = cv2.medianBlur(otsu, 3)
    return otsu


def show_image(image, title):
    """
    Function to show an image to the screen
    :param image: image to be shown - np array
    :param title: window title - string
    :return: None
    """
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def make_positive_point(point, img):
    """
    Transform a point to fit in an image
    :param point: point
    :param img: image
    :return: point transformed
    """
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0

    return point


def save_image_ext(img, path, name, extension):
    """
    Function to save an image to a given location
    :param img: image to save - np array
    :param path: path to the directory - string
    :param name: image name - string
    :param extension: extension of the image - string
    :return: None
    """
    cv2.imwrite(path + "\\" + name + "." + extension, img)


def save_image(img, path):
    """
    Function to save an image to a given location
    :param img: image to save - np array
    :param path: path to the directory - string
    :return: None
    """
    cv2.imwrite(path, img)


def get_hist_rgb(image, draw=True):
    """
    Function to get the rgb histogram of rgb image
    :param image: image to get histogram from
    :param draw: if draw is True the histogram will be plotted using matplotlib
    :return: - list containing histogram for every color
    """
    hist = []
    color = ('b', 'g', 'r')
    for channel, col in enumerate(color):
        histr = cv2.calcHist([image], [channel], None, [256], [0, 256])
        hist.append(histr)
        if draw is True:
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
    if draw is True:
        plt.title('Historgram for color scale picture')
        plt.show()

    return hist


def get_hist_gr_scale(image, draw=True):
    """
    Function to get the grayscale histogram of an image
    :param image: image to get histogram from
    :param draw: if draw is True the histogram will be plotted using mat plot lib
    :return: - list containing histogram for the image
    """

    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    if draw:
        plt.hist(image.ravel(), 256, [0, 256])
        plt.show()
    return hist


def read_json(file):
    """
    Function to read the date from a json file
    :param file: filename to be read
    """
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def intersect(D, B, D1, B1):
    """
    Function to check if two rectangles intersect
    :param D: -left down corner of rectangle1
    :param B: -right up corner of rectangle1
    :param D1: -left down corner of rectangle2
    :param B1: -right up corner of rectangle2
    :return: True if the rectangles overlap false otherwise
    """
    self_top_right_x = B[0]
    self_top_right_y = B[1]
    self_bottom_left_x = D[0]
    self_bottom_left_y = D[1]
    other_top_right_x = B1[0]
    other_top_right_y = B1[1]
    other_bottom_left_x = D1[0]
    other_bottom_left_y = D1[1]
    return not (
                self_top_right_x <= other_bottom_left_x or self_bottom_left_x >= other_top_right_x or self_top_right_y >=
                other_bottom_left_y or self_bottom_left_y <= other_top_right_y)


def unite_rectangles(rectangles):
    """
    Function to unite a set of rectangles
    :param rectangles: -rectangles to be united
    :return: the union of rectangles
    """
    D, B = rectangles[0]
    x_min = D[0]
    x_max = B[0]
    y_min = B[1]
    y_max = D[1]
    for i in range(1, len(rectangles)):
        D, B = rectangles[i]
        if D[0] < x_min:
            x_min = D[0]
        if B[0] > x_max:
            x_max = B[0]
        if D[1] > y_max:
            y_max = D[1]
        if B[1] < y_min:
            y_min = B[1]
    return ([x_min, y_max], [x_max, y_min])


def save_txt_file(path, name, rois_coords):
    """
    Function to save a list of rectangles
    :param path: - path where to save the file
    :param name: - name of the image
    :param roi_coords: - rois coordinates
    """
    filename = path + "\\rois.txt"
    text_to_save = ""
    for coord in rois_coords:
        text_to_save += coord[0][0] + " " + str(coord[1]) + " " + str(coord[2]) + " " + str(coord[3]) +  " " + \
                        str(coord[4]) + ","
    text_to_save = text_to_save[:-1]
    with open(filename, 'a') as f:
        line = name + " " + text_to_save
        f.write(line + "\n")


def key_of_dict_val(map_, value):
    """
    Function to return  the key of a value in a map
    :param map_: dictionary
    :param value: some value in the dictionary
    :return: None if there is no key with the given value, the key otherise
    """

    for key, val in map_.items():
        if val == value:
            return key
    return None


def binarization(img, resize_ratio, reverse_binarization, second_processing, rois, vertical):
    """
    Function to binarize an image
    :param img: image to be binarized
    :param resize_ratio: resize ratio
    :param reverse_binarization: True if we want inverse binarization False otherwise
    :param second_processing: True for second processing, False otherwise
    :param rois:
    :param vertical:
    :return: the binarized image
    """
    rows, cols = img.shape
    if rows * cols != 0:
        valueOfBrightness = cv2.sumElems(img)[0] / (rows * cols)
    do_hist_eq = False
    if rois:
        do_hist_eq = True
        if valueOfBrightness < 30:
            for i in range(rows):
                for j in range(cols):
                    if img[i, j] > 20:
                        if img[i, j] + 100 > 255:
                            img[i,j] = 255
                        else:
                            img[i, j] += 100
        elif valueOfBrightness >= 30 and valueOfBrightness < 40:
            for i in range(rows):
                for j in range(cols):
                    if img[i, j] > valueOfBrightness + 10:
                        if img[i, j] + 100 > 255:
                            img[i,j] = 255
                        else:
                            img[i, j] += 100
    img = cv2.erode(img, (3, 3), 30)
    if reverse_binarization:
        img = 255 - img
    img = cv2.resize(img, (int(img.shape[1] * resize_ratio), int(img.shape[0] * resize_ratio)))
    th_otsu = morphological_transformation(img, second_processing, reverse_binarization, rois, do_hist_eq, vertical)
    if reverse_binarization:
        th_otsu = 255 - th_otsu
    return th_otsu


def sq_distance_points(A, B):
    """
    Function which calculates the squared distance between two poitns
    :param A: First point
    :param B: Second point
    :return: Squared distance between A and B
    """
    return (A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2

def morphological_transformation(img, second_processing, reverse_binarization, rois, histogramEq, vertical):
    """
    Morphological transformations for binarization
    :param img: image to be processed
    :param second_processing: True for second_processing, false otherwise
    :param reverse_binarization: True if we want reverse binarization, False otherwise
    :param rois:
    :param histogramEq: True if we want histogram equalization, false otherwise
    :param vertical:
    :return: the processed image
    """
    if second_processing:
        if reverse_binarization:
            kernel_for_erosion = np.ones((3+3, 3+3), np.uint8)
            img = cv2.erode(img, kernel_for_erosion, iterations=1)
        else:
            kernel_for_dilation = np.ones((3+3, 3+3), np.uint8)
            img = cv2.dilate(img, kernel_for_dilation, iterations=1)
    if rois:
        if histogramEq:
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(20, 20))
            img = clahe.apply(img)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    ret3, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if not rois:
        kernel_for_erosion = np.ones((3, 3), np.uint8)
        img = cv2.erode(img, kernel_for_erosion, iterations=1)
        kernel_for_dilation = np.ones((3, 3), np.uint8)
        img = cv2.dilate(img, kernel_for_dilation, iterations=1)
    return img

# name, img = read_image(r'C:\Users\BDG3CLJ\Desktop\CRAFT-Operations\first_processed\p01_ebs3-1-label-benz2-010-ok.jpg')
# #apelare functie cu reverse_binarization == True
# img = binarization(img, 1, True, False, True, False)
# show_image(img, '')

def clean_dirs(*args):
    """
    Function to delete the content of the given folders
    :param args: folders
    :return: None
    """
    for arg in args:
        try:
            shutil.rmtree(arg, ignore_errors=False, onerror=None)
            os.mkdir(arg)
        except:
            print("This folder doesn't exist.")

def remove_logo(logo, image, too_dark):
    """
    Function to remove the logo from an image
    :param logo: logo to be removed
    :param image: image to remove logo from
    :param too_dark: tells if the image needs to be illuminated
    :return: the new image
    """
    to_ret = image.copy()
    if too_dark:
        image = cv2.convertScaleAbs(image, alpha=2.2, beta=50)
    method = cv2.TM_SQDIFF_NORMED
    result = cv2.matchTemplate(logo, image, method)
    mn, _, mnLoc, _ = cv2.minMaxLoc(result)
    Mpx, Mpy = mnLoc
    trows, tcols = logo.shape[:2]
    to_ret[Mpy : Mpy + trows, Mpx : Mpx + tcols] = 0
    return to_ret