import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time


dialation_size = 1
th = 300

def load_image(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def get_es_roi(image):
    x = image.shape[0]
    y = image.shape[1]
    new_image = image[int(x/3 + x/40): -int(x/4), int(y/2 + y/32): -int(y/4 + y/30), :]
    return new_image


def get_thresh(image):
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]
    R_ret, R_thresh = cv2.threshold(R, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    G_ret, G_thresh = cv2.threshold(G, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    B_ret, B_thresh = cv2.threshold(B, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return R_thresh & G_thresh & B_thresh


def erode(image, erosion_size):
    element = cv2.getStructuringElement(1, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))
    erosion_dst = cv2.erode(image, element)
    return erosion_dst


def dialate(image):
    element = cv2.getStructuringElement(1, (2 * dialation_size + 1, 2 * dialation_size + 1),
                                       (dialation_size, dialation_size))
    dialation_dst = cv2.dilate(image, element)
    return dialation_dst


def contours(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image = cv2.drawContours(image, contours, -1, (0,255,0), 15)   


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
        if np.sum(arr) > sums[-5]:
            image = image + arr
    return image.astype('uint8')


def find_thermo(image):
    output = cv2.connectedComponentsWithStats(
	    image, 8, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    arr = np.array((labels==6).astype(int))
    thermos = []
    for i in range(max(map(max, labels))):
        if i== 0:
            continue
        arr = np.array((labels==i).astype(int))
        if 2.1 < stats[i][2] / stats[i][3] < 2.3 and np.sum(arr) > 3000:
            thermos.append(stats[i])
    if len(thermos) == 1:
        return thermos[0]
    else:
        print("length of thermos is more than 1")
    

def get_thermo(image, ranges):
    rm = 10
    image = image[ranges[1]+rm:ranges[1]+ranges[3]-rm, ranges[0]+rm:ranges[0]+ranges[2]-rm, :]
    return image


def get_half_thermo(image):
    x = image.shape[0]
    y = image.shape[1]
    return image[int(x/2) + 6: -10,:]


def static_thresh(image, mi):
    ret, thresh = cv2.threshold(image, mi,255,cv2.THRESH_BINARY)
    return thresh

def static_thresh_INV(image):
    ret, thresh = cv2.threshold(image, 0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return thresh

def get_canny(img):
    canny =  cv2.Canny(img, 60, 250)
    return canny


def find_thermo_exact(image):
    first = 13
    second = 172
    return image[:, first:second]
    

def get_arrow(image):
    output = cv2.connectedComponentsWithStats(
	    image, 8, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    arr = np.array((labels==6).astype(int))
    arrows = []
    for i in range(max(map(max, labels))+1):
        if i== 0:
            continue
        arr = np.array((labels==i).astype(int))
        # plt.imshow(arr)
        # plt.show()
        if 0.3 < stats[i][2] / stats[i][3] < 0.8 and np.sum(arr) > 50 and 0 <= stats[i][1] < 20:
            # plt.imshow(arr)
            # plt.show()
            arrows.append(arr)
    return arrows


def get_arrow_head(arrow):
    indexes = []
    for i in arrow:
        for ind, j in enumerate(i):
            if j == 1:
                while i[ind] == 1:
                    indexes.append(ind)
                    ind = ind + 1
                return np.sum(indexes) / len(indexes)    


def get_theta(num, immax):
    n0 = 6.2
    n5 = 11.7
    n10 = 17
    n15 = 22.6
    n20 = 29
    n25 = 34.8
    n30 = 41.5
    n35 = 47.9
    n40 = 54.6
    n45 = 61.7
    n50 = 68.7
    n55 = 76.1
    n60 = 83.5
    n65 = 90.9
    n70 = 97.3
    n75 = 104.0
    n80 = 110.7
    n85 = 117.5
    n90 = 123.9
    n95 = 129.3
    n100 = 137.7
    n110 = 147.0
    n120 = 154
    def line(x1,x2,y1,y2,num):
        return ((((y2 - y1)/(x2-x1))*(num - x1))+y1)

    if num < n0:
        return line(0,n0,-10,0,num)

    if n0 <= num < n5:
        return line(n0,n5,0,5,num) 
    
    if n5 <= num < n10:
        return line(n5,n10,5,10,num) 
    
    if n10 <= num < n15:
        return line(n10,n15,10,15,num) 
    
    if n15 <= num < n20:
        return line(n15,n20,15,20,num) 
    
    if n20 <= num < n25:
        return line(n20,n25,20,25,num) 
    
    if n25 <= num < n30:
        return line(n25,n30,25,30,num) 
    
    if n30 <= num < n35:
        return line(n30,n35,30,35,num) 
    
    if n35 <= num < n40:
        return line(n35,n40,35,40,num) 
    
    if n40 <= num < n45:
        return line(n40,n45,40,45,num) 
    
    if n45 <= num < n50:
        return line(n45,n50,45,50,num) 
    
    if n50 <= num < n55:
        return line(n50,n55,50,55,num) 
    
    if n55 <= num < n60:
        return line(n55,n60,55,60,num) 
    
    if n60 <= num < n65:
        return line(n60,n65,60,65,num) 
    
    if n65 <= num < n70:
        return line(n65,n70,65,70,num) 
    
    if n70 <= num < n75:
        return line(n70,n75,70,75,num) 
    
    if n75 <= num < n80:
        return line(n75,n80,75,80,num) 
    
    if n80 <= num < n85:
        return line(n80,n85,80,85,num) 
    
    if n85 <= num < n90:
        return line(n85,n90,85,90,num) 

    if n90 <= num < n95:
        return line(n90,n95,90,95,num) 

    if n95 <= num < n100:
        return line(n95,n100,95,100,num) 
    
    if n100 <= num < n110:
        return line(n100,n110,100,110,num) 
    
    if n110 <= num < n120:
        return line(n110,n120,110,120,num) 

    if n120 <= num < immax:
        return line(n120,immax,120,140,num) 
     

def my_static_thresh(image):
    i = 100
    while(True):
        thresh = static_thresh(image, i)
        thresh = erode(thresh,3)
        # plt.imshow(thresh)
        # plt.show()
        arrow = get_arrow(thresh)
        # plt.imshow(arrow)
        # plt.show()
        if len(arrow) == 1:
            return arrow[0]
        elif i == 225:
            return None
        else:
            i = i + 5 

def show_number_on_image(img, number):
    cv2.putText(img=img, text=str(number), org=(int(img.shape[1] * 0.50), int(img.shape[0] * 0.90)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(255, 0, 0),thickness=1)
    plt.imshow(img)
    plt.show()

def main():
    # im = str(sys.argv[1])
    im = "./two.jpg"
    try:
        image = load_image(im)
        es_image = get_es_roi(image)
        gray = get_gray(es_image)
        image = static_thresh_INV(gray)
        image = erode(image, 1)
        image = remove_small_elements(image)
        ranges = find_thermo(image)
        thermo = get_thermo(es_image, ranges)
        gray = get_gray(thermo)
        gray = find_thermo_exact(gray)
        gray = get_half_thermo(gray)
        histr = cv2.calcHist([gray],[0],None,[256],[0,256])

        # show the plotting graph of an image
        arrow = my_static_thresh(gray)
        arrow_head = get_arrow_head(arrow)
        theta = get_theta(arrow_head, gray.shape[1])
        print(theta)
        show_number_on_image(thermo, theta)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()