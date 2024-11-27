import glob
import numpy as np
import cv2
import time 

def NCC(image1, image2):

    image1_array = np.array(image1)
    image2_array = np.array(image2)
    # Flatten the arrays to 1D arrays of pixel values
    image1_pixels = image1_array.flatten()
    image2_pixels = image2_array.flatten()
    norm_corr_coeff = np.corrcoef(image1_pixels, image2_pixels)[0, 1] / (np.std(image1_pixels) * np.std(image2_pixels))
    print(norm_corr_coeff)
    return norm_corr_coeff
# ------------------ Normalised Cross Correlation ------------------ #
def Normalised_Cross_Correlation(roi, target):
    # Normalised Cross Correlation Equation
    cor = np.sum(roi * target)
    nor = np.sqrt( (np.sum(roi ** 2))) * np.sqrt(np.sum(target ** 2))
   
    
    return cor / nor

# ----------------------- template matching ----------------------- #
def template_matching(img, target):
    # initial parameter
    height, width, _ = img.shape
    tar_height, tar_width, _ = target.shape

    (max_Y, max_X) = (0, 0)
    MaxValue = 0

    # Set image, target and result value matrix
    img = np.array(img, dtype="int")
    target = np.array(target, dtype="int")
    NccValue = np.zeros((height-tar_height, width-tar_width))

    # calculate value using filter-kind operation from top-left to bottom-right
    for y in range(0, height-tar_height):
        for x in range(0, width-tar_width):
            # image roi
            roi = img[y : y+tar_height, x : x+tar_width]
            # calculate ncc value
            NccValue[y, x] = Normalised_Cross_Correlation(roi, target)
            # find the most match area
            if NccValue[y, x] > MaxValue:
                MaxValue = NccValue[y, x]
                (max_Y, max_X) = (y, x)

    return (max_X, max_Y)


# -------------------------- main -------------------------- #
if __name__ == '__main__':
    # Read image and target(template)
    circle_path = glob.glob(r'./circle/*')
    cross_path = glob.glob(r'./cross/*')

    pattern_path = glob.glob(r'./pattern/*') 
    image = cv2.imread(circle_path[0])
    target = cv2.imread(pattern_path[2])
    
    height, width, _ = target.shape



    start = time.time()
    # function
    top_left = template_matching(image, target)
    # draw rectangle on the result region
    cv2.rectangle(image, top_left, (top_left[0] + width, top_left[1] + height), (0,0,255), 2)

    end = time.time()

    print("執行時間：%f 秒" % (end - start))
    cv2.imshow('Template_cross Matching', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
