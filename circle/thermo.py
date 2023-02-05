import cv2
import numpy as np
import sys
import math
import matplotlib.pyplot as plt

SHOW_NUMBER = True
DRAW_LINES = False

def load_image(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def get_one_circles(img, min_r, max_r):
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=min_r, maxRadius=max_r)
    if circles is None:
        print("No circle detected")
        sys.exit()
    circle_x, circle_y, circle_r = circles[0][0]
    circle_x = int(circle_x)
    circle_y = int(circle_y)
    circle_r = int(circle_r)
    return circle_x, circle_y, circle_r
    

def get_rio(img, circle_x, circle_y, circle_r):
    mask1 = np.zeros(img.shape)
    mask2 = np.zeros(img.shape)
    mask1 = cv2.circle(mask1, (circle_x, circle_y), circle_r-30, (1,1,1), -1).astype(np.uint8)
    mask2 = cv2.circle(mask2, (circle_x, circle_y), circle_r-31, (1,1,1), -1).astype(np.uint8)
    img = np.multiply(img, mask1)
    img = img[circle_y - circle_r:circle_y +
                        int(circle_r * 0.5), circle_x - circle_r:circle_x + circle_r]
    mask2 = mask2[circle_y - circle_r:circle_y +
                        int(circle_r * 0.5), circle_x - circle_r:circle_x + circle_r]
    return img, mask2

def get_show_rio(img, circle_x, circle_y, circle_r):
    mask1 = np.zeros(img.shape)
    mask1 = cv2.circle(mask1, (circle_x, circle_y), circle_r, (1,1,1), -1).astype(np.uint8)
    img = np.multiply(img, mask1)
    img = img[circle_y - circle_r:circle_y +
                        circle_r, circle_x - circle_r:circle_x + circle_r]
    return img



def get_gray_thresh(img):
    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th2


def central_circle_mask(gray_roi):
    circle1_x, circle1_y, circle1_r = get_one_circles(gray_roi, min_r=20, max_r=40)
    mask = np.ones(gray_roi.shape)
    mask = cv2.circle(mask, (circle1_x, circle1_y), circle1_r + 5, 0, -1).astype(np.uint8)
    return mask


def get_canny(img, mask):
    canny =  cv2.Canny(img, 40, 300)
    return np.multiply(canny, mask)


def remove_central_circle(img, mask):
    return np.multiply(img, mask)


def remove_small_elements(img):
    output = cv2.connectedComponentsWithStats(
	    img, 8, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    sums = []
    for i in range(max(map(max, labels))):
        if i== 0:
            continue
        arr = np.array((labels==i).astype(int))
        sums.append(np.sum(arr))
    sums.sort()
    image = np.zeros(labels.shape)
    for i in range(max(map(max, labels))):
        if i== 0:
            continue
        arr = np.array((labels==i).astype(int))
        if np.sum(arr) > sums[-4]:
            image = image + arr
    return image


def get_lines(img, num_lines):
    img = img.astype(np.uint8)
    lines = 0
    i = 100
    while(True):
        lines = cv2.HoughLines(img, 1, np.pi/360, i)
        cv2.HoughLinesP(img,rho = 1,theta = 1*np.pi/180,threshold = i, maxLineGap = 500)
        try:
            if len(lines) > num_lines:
                break
        except:
            pass
        i = i - 1
    return lines


def get_line_mask(lines, shape):
    image = np.zeros(shape)
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        image = cv2.line(image, pt1, pt2, 1, 5, cv2.LINE_AA)
    return image.astype(np.uint8)


def get_line_side(canny, line_mask):
    hand = np.multiply(canny, line_mask)
    left = hand[:, :100]
    right = hand[:, 100:]
    left = np.sum(left)
    right = np.sum(right)
    if right > left:
        return True
    else:
        return False


def get_number(deg, is_right):
    deg_b = deg
    deg = (np.pi/2) - deg
    if deg == 0:
        if is_right == False:
            return 0 
        else:
            return 100
    
    if deg < 0:
        num = ((50/(np.pi/2)) * (abs(deg) - (np.pi/2))) + 50
        if is_right == True:
            num = num + 100
        return num

    if deg > 0:
        num = ((50/(np.pi/2)) * deg_b) + 50
        if is_right == False:
            num = -(100 - num)
        return num


def get_theta(lines):
    thetas = [i[0][1] for i in lines]
    sum = np.sum(thetas)
    theta = sum / len(thetas)
    return theta


def draw_lines(img, lines):
    for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            img = cv2.line(img, pt1, pt2, 1, 1, cv2.LINE_AA)

def show_number_on_image(img, number):
    cv2.putText(img=img, text=str(number), org=(0, int(img.shape[1] * 0.75)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 0, 0),thickness=2)
    plt.imshow(img)
    plt.show()


def main():
    im = "./one.jpg"
    # im = str(sys.argv[1])

    image = load_image(im)
    gray_image = get_gray(image)
    circle1_x, circle1_y, circle1_r = get_one_circles(gray_image, min_r=100, max_r=200)
    image_roi, image_mask = get_rio(image, circle1_x, circle1_y, circle1_r)
    gray_roi, gray_mask = get_rio(gray_image, circle1_x, circle1_y, circle1_r)
    mask = central_circle_mask(gray_roi)
    canny = get_canny(gray_roi, gray_mask)
    canny = remove_central_circle(canny, mask)
    canny = remove_small_elements(canny)
    lines = get_lines(canny, 0)
    line_mask = get_line_mask(lines, canny.shape)
    is_right = get_line_side(canny, line_mask)
    theta = get_theta(lines)     
    number = get_number(theta, is_right)
    print(number)
    
    if DRAW_LINES:
        draw_lines(image_roi, lines)
    if SHOW_NUMBER:
        image_show_roi = get_show_rio(image, circle1_x, circle1_y, circle1_r)
        show_number_on_image(image_show_roi, number)




if __name__ == "__main__":
    main()