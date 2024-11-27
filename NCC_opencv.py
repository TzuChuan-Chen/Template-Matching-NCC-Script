import cv2
from matplotlib import pyplot as plt
import time
import glob

def template_matching(img, target):

    img = cv2.medianBlur(img, 15)
    # copy image in order to compare with different method
    img2 = img.copy()
    
    w, h = target.shape[::-1]

    # All the 6 methods for comparison in a list
    methods = eval('cv2.TM_CCOEFF_NORMED')

    # do template matching with different method
    img = img2.copy()

    # Apply template Matching
    res = cv2.matchTemplate(img,target,methods)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)


    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)


    return top_left, bottom_right


# -------------------------- main -------------------------- #
if __name__ == '__main__':
    # Read image and target(template)
    circle_path = ['./circle\\Panel1_circle1.bmp', './circle\\Panel1_circle2.bmp', './circle\\Panel1_circle3.bmp', './circle\\Panel1_circle4.bmp', './circle\\Panel2_circle1.bmp', './circle\\Panel2_circle2.bmp', './circle\\Panel2_circle3.bmp', './circle\\Panel2_circle4.bmp', './circle\\Panel3_circle1.bmp', './circle\\Panel3_circle2.bmp', './circle\\Panel3_circle3.bmp', './circle\\Panel3_circle4.bmp', './circle\\Panel4_circle1.bmp', './circle\\Panel4_circle2.bmp', './circle\\Panel4_circle3.bmp', './circle\\Panel4_circle4.bmp']
    cross_path = ['./cross\\Panel1_cross1.bmp', './cross\\Panel1_cross2.bmp', './cross\\Panel1_cross3.bmp', './cross\\Panel1_cross4.bmp', './cross\\Panel2_cross1.bmp', './cross\\Panel2_cross2.bmp', './cross\\Panel2_cross3.bmp', './cross\\Panel2_cross4.bmp', './cross\\Panel3_cross1.bmp', './cross\\Panel3_cross2.bmp', './cross\\Panel3_cross3.bmp', './cross\\Panel3_cross4.bmp', './cross\\Panel4_cross1.bmp', './cross\\Panel4_cross2.bmp', './cross\\Panel4_cross3.bmp', './cross\\Panel4_cross4.bmp']

    pattern_path = ['./pattern\\Template_BorderCircle.bmp', './pattern\\Template_BorderCross.bmp', './pattern\\Template_circle.bmp', './pattern\\Template_cross.bmp']

    ### Image = circle
    ### Template = Template_BorderCross.bmp
    for i in circle_path:
        image = cv2.imread(i)
        target_Border = cv2.imread(pattern_path[0])
        target_circle = cv2.imread(pattern_path[2])
        
        _, w_Border, h_Border = target_Border.shape[::-1]
        _, w_circle, h_circle = target_circle.shape[::-1]

        start = time.time()
        # convert to grayscale image
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        target_Border = cv2.cvtColor(target_Border, cv2.COLOR_BGR2GRAY)
        target_circle = cv2.cvtColor(target_circle, cv2.COLOR_BGR2GRAY)
        top_left_bor, bottom_right_bor = template_matching(image_gray, target_Border)
        top_left_cir, bottom_right_cir = template_matching(image_gray, target_circle)

        # draw the match region
        cv2.rectangle(image, top_left_bor, bottom_right_bor, (0, 0, 255), 2)
        cv2.circle(image, (top_left_bor[0]+(w_Border//2), top_left_bor[1]+(h_Border//2)), 1, (0, 0, 255),4)
        cv2.line(image, (top_left_bor[0], top_left_bor[1]+(h_Border//2)), (bottom_right_bor[0],  top_left_bor[1]+(h_Border//2)), (0,0,255), 1)
        cv2.line(image, (top_left_bor[0]+(w_Border//2), top_left_bor[1]), (top_left_bor[0]+(w_Border//2),  bottom_right_bor[1]), (0,0,255), 1)
        print('Border',top_left_bor[0]+(w_Border//2), top_left_bor[1]+(h_Border//2))

        cv2.rectangle(image, top_left_cir, bottom_right_cir, (255, 0, 0), 2)
        cv2.circle(image, (top_left_cir[0]+(w_circle//2), top_left_cir[1]+(h_circle//2)), 1, (255, 0, 0),4)
        cv2.line(image, (top_left_cir[0], top_left_cir[1]+(h_circle//2)), (bottom_right_cir[0],  top_left_cir[1]+(h_circle//2)), (255,0,0), 1)
        cv2.line(image, (top_left_cir[0]+(w_circle//2), top_left_cir[1]), (top_left_cir[0]+(w_circle//2),  bottom_right_cir[1]), (255,0,0), 1)
        print('circle',top_left_cir[0]+(w_circle//2), top_left_cir[1]+(h_circle//2))
        end = time.time()
        print(f'{i[-18:-4]}_{pattern_path[0][-16:-4]}.bmp', "的執行時間：%f 秒" % (end - start))
        cv2.imwrite(f'result/{i[-18:-4]}_{pattern_path[0][-16:-4]}.bmp',image)


    ### Image = cross, 
    ### Template = Template_Cross.bmp
    for i in cross_path:
        image = cv2.imread(i)
        target_Border = cv2.imread(pattern_path[1])
        target_circle = cv2.imread(pattern_path[3])
        
        _, w_Border, h_Border = target_Border.shape[::-1]
        _, w_circle, h_circle = target_circle.shape[::-1]

        start = time.time()
        # convert to grayscale image
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        target_Border = cv2.cvtColor(target_Border, cv2.COLOR_BGR2GRAY)
        target_circle = cv2.cvtColor(target_circle, cv2.COLOR_BGR2GRAY)
        top_left_bor, bottom_right_bor = template_matching(image_gray, target_Border)
        top_left_cir, bottom_right_cir = template_matching(image_gray, target_circle)

        # draw the match region
        cv2.rectangle(image, top_left_bor, bottom_right_bor, (0, 0, 255), 2)
        cv2.circle(image, (top_left_bor[0]+(w_Border//2), top_left_bor[1]+(h_Border//2)), 1, (0, 0, 255),4)
        cv2.line(image, (top_left_bor[0], top_left_bor[1]+(h_Border//2)), (bottom_right_bor[0],  top_left_bor[1]+(h_Border//2)), (0,0,255), 1)
        cv2.line(image, (top_left_bor[0]+(w_Border//2), top_left_bor[1]), (top_left_bor[0]+(w_Border//2),  bottom_right_bor[1]), (0,0,255), 1)
        print('Border',top_left_bor[0]+(w_Border//2), top_left_bor[1]+(h_Border//2))

        cv2.rectangle(image, top_left_cir, bottom_right_cir, (255, 0, 0), 2)
        cv2.circle(image, (top_left_cir[0]+(w_circle//2), top_left_cir[1]+(h_circle//2)), 1, (255, 0, 0),4)
        cv2.line(image, (top_left_cir[0], top_left_cir[1]+(h_circle//2)), (bottom_right_cir[0],  top_left_cir[1]+(h_circle//2)), (255,0,0), 1)
        cv2.line(image, (top_left_cir[0]+(w_circle//2), top_left_cir[1]), (top_left_cir[0]+(w_circle//2),  bottom_right_cir[1]), (255,0,0), 1)
        print('cross',top_left_cir[0]+(w_circle//2), top_left_cir[1]+(h_circle//2))
        end = time.time()
        print(f'{i[-18:-4]}_{pattern_path[0][-16:-4]}.bmp', "的執行時間：%f 秒" % (end - start))
        cv2.imwrite(f'result/{i[-18:-4]}_{pattern_path[0][-16:-4]}.bmp',image)
        # cv2.imshow(f'Image: {i} Template: {pattern_path[0]}', image)
        
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    