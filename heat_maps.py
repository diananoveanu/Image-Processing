import cv2
import numpy as np

"""
Function to cut an image in half
:param image: image to be cut
:return: -left half of the image 
"""
def crop_half_left(image):
    cropped_img = image[:, :image.shape[1]//2]
    return cropped_img


"""
Function to cut an image in half
:param image: image to be cut
:return: -right half of the image 
"""
def crop_half_right(image):
    cropped_img = image[:, image.shape[1]//2:]
    return cropped_img
    
    
"""
Function to get heat zones from the original image
:param heat: -heat zones image
:param orig_image: -original image
:return: heat zones of the image, scaled to the original image
"""
def get_heat_zones(heat, orig_img):
    
    img = heat
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
    mask3 = cv2.inRange(hsv, np.array([0,100,100]), np.array([70,255,255]));
    mask= mask1 | mask2 | mask3
    #mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(img, img, mask=mask)
    res = crop_half(res)
    if len(orig_img.shape) == 3:
        orig_h, orig_w, _ = orig_img.shape
    else:
        orig_h, orig_w = orig_img.shape
    x_factor = orig_w 
    y_factor = orig_h
    res = cv2.resize(res, (x_factor, y_factor), interpolation = cv2.INTER_AREA)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    ret,res = cv2.threshold(res,0,255,cv2.THRESH_BINARY)
    return res