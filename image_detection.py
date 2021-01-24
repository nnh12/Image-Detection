import cv2
import numpy as np
from matplotlib import pyplot as plt

def get_image(location):
    image = cv2.imread(location)
    return image

def get_template(location):
    template = cv2.imread(location)
    return template

def get_height(template):
    return template.shape[0]

def get_width(template):
    return template.shape[1]

def template_match_topLeft_one(image, template, mask):
    match = cv2.matchTemplate(image, template, mask)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
    top_left = max_loc
    return top_left[1]
    
def template_match_topLeft_zero(image, template, mask):
    match = cv2.matchTemplate(image, template, mask)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
    top_left = max_loc
    return top_left[0]

def template_match_bottomright_one(image, template, mask):
    match = cv2.matchTemplate(image, template, mask)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
    top_left = max_loc
    bottom_right = (top_left[0]+ get_width(template), top_left[1]+ get_height(template))
    return bottom_right[1]

def template_match_bottomright_zero(image, template, mask):
    match = cv2.matchTemplate(image, template, mask)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
    top_left = max_loc
    bottom_right = (top_left[0]+ get_width(template), top_left[0]+ get_height(template))
    return bottom_right[0]

def template_match_topleft(image, template, mask):
    match = cv2.matchTemplate(image, template, mask)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
    return max_loc

def template_match_bottomright(image, template, mask):
    match = cv2.matchTemplate(image, template, mask)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
    top_left = max_loc
    return (top_left[0]+ get_width(template), top_left[1]+ get_height(template))

def coordinates(image, template, mask):
    top_left_one = template_match_topLeft_one(image, template, mask)
    print(top_left_one)
    top_left_zero = template_match_topLeft_zero(image, template, mask)
    print(top_left_zero)
    bottom_right_one = template_match_bottomright_one(image, template, mask)
    print(bottom_right_one)
    bottom_right_zero = template_match_bottomright_zero(image, template, mask)
    print(bottom_right_zero)
    imCropped = image[top_left_one+50: top_left_zero -100, bottom_right_one + 100: bottom_right_zero]
    return imCropped

def plot(picture,crop):
    plt.subplot(121),plt.imshow(picture)
    plt.title('Original Pictures'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(coordinates(picture,crop, cv2.TM_CCOEFF))
    plt.show()

image = get_image('C:\\Users\\natha\\Desktop\\cwrubotics\\MateROV\\rov-vision\\Practice\\nba.jpg')
template = get_template('C:\\Users\\natha\\Desktop\\cwrubotics\\MateROV\\rov-vision\\Practice\\Face.png')
plot(image, template)




