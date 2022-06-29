import sys
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import numpy as np
import json
import math
import random
from functools import cmp_to_key
# loading parameter jason file

BASE_PATH = "/home/user/frinks/python_backend"

data_jsonx = json.load(open(f"{BASE_PATH}/scripts/deviation_data.json",))[0]

# functon to add 50 pixels to the top of image


def add_black(img):
    img_h, img_w, c = img.shape
    black = np.zeros((50+img_h, img_w, 3))
    black[50:, :, :] = img
    return black


def letter_cmp(a, b):
    if a[0] > b[0]:
        return 1
    elif a[0] == b[0]:
        if a[1] > b[1]:
            return -1
        else:
            return 1
    else:
        return -1


letter_cmp_key = cmp_to_key(letter_cmp)


def letter_cmp2(a, b):
    if a[0] > b[0]:
        return 1
    elif a[0] == b[0]:
        if a[1] > b[1]:
            return -1
        else:
            return 1
    else:
        return -1


letter_cmp_key2 = cmp_to_key(letter_cmp2)


def func(img2, template, template_name, cus_th=data_jsonx["threshold"]):
    result = cv2.matchTemplate(img2, template, cv2.TM_CCOEFF_NORMED)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
    (startX, startY) = maxLoc
    endX = startX + template.shape[1]
    endY = startY + template.shape[0]
    img3 = img2.copy()
    cv2.rectangle(img3, (startX, startY), (endX, endY), (255, 0, 0), 2)
    cv2.putText(img3, template_name, (startX, startY-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    # show the output image
    # plt.imshow(result)
    yes = "True"
    # plt.show()
    # plt.imshow(img3)#"Output", img3)
    # plt.show()
    # cv2.imwrite("template_matching_image_without_key.jpg",img3)
    result = cv2.matchTemplate(img2, template, cv2.TM_CCOEFF_NORMED)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
    (startX, startY) = maxLoc
    endX = startX + template.shape[1]
    endY = startY + template.shape[0]
    img3 = img2.copy()
    ##cv2.rectangle(img3, (startX, startY), (endX, endY), (255, 0, 0), 3)
    # show the output image
    # plt.imshow(result)
    # plt.show()
    # plt.imshow(img3)#"Output", img3)
    # plt.show()
    # cv2.imwrite("template_matching_image_with_key.jpg",img3)
    # ------------------------------< IMPORANT PARAMETER
    # threshold = data_jsonx["threshold"]
    threshold = cus_th
    (yCoords, xCoords) = np.where(result >= threshold)
    img3 = img2.copy()
    #print("[INFO] {} matched locations *before* NMS".format(len(yCoords)))
    # loop over our starting (x, y)-coordinates
    tW = template.shape[1]
    tH = template.shape[0]
    # for (x, y) in zip(xCoords, yCoords):
    # draw the bounding box on the image
    # cv2.rectangle(clone, (x, y), (x + tW, y + tH),
    # (255, 0, 0), 3)
    # show our output image *before* applying non-maxima suppression
    ##cv2.imshow("Before NMS", clone)
    # cv2.waitKey(0)
    # cv2.imwrite("template_matching_image_with_key_before_NMS"+str(threshold)+".jpg",clone)
    rects = []
    # loop over the starting (x, y)-coordinates again
    for (x, y) in zip(xCoords, yCoords):
        # update our list of rectangles
        rects.append((x, y, x + tW, y + tH))
    # apply non-maxima suppression to the rectangles
    pick = non_max_suppression(np.array(rects))
    #print("[INFO] {} matched locations *after* NMS".format(len(pick)))
    # loop over the final bounding boxes
    #img3 = img2.copy()
    for (startX, startY, endX, endY) in pick:
        # draw the bounding box on the image
        cv2.rectangle(img3, (startX, startY), (endX, endY),
                      (255, 0, 0), 3)
    if len(pick) == 0:
        yes = "False"
        return img3, threshold, yes
    # show the output image
    # plt.imshow(img3)#"template_matching_image_with_key_after_NMS"+str(threshold)+".jpg", img3)
    # plt.show()
    result = cv2.matchTemplate(img2, template, cv2.TM_CCOEFF_NORMED)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
    (startX, startY) = maxLoc
    endX = startX + template.shape[1]
    endY = startY + template.shape[0]
    img3 = img2.copy()
    cv2.rectangle(img3, (startX, startY), (endX, endY), (255, 0, 0), 2)
    cv2.putText(img3, template_name, (startX, startY-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    a = [startX, startY, endX, endY]
    return img3, threshold, yes, a

    #cv2.imwrite("template_matching_image_with_key_after_NMS"+str(threshold)+".jpg", img3)


# main
# sys.argv[1]
param_1 = f'{BASE_PATH}/scripts/2000.bmp'
# sys.argv[2]
param_2 = f'{BASE_PATH}/scripts/2000.json'
param_3 = sys.argv[1]
img1 = cv2.imread(param_1, cv2.IMREAD_COLOR)
f = open(param_2,)
# cv2.imshow('a',img1)
# cv2.waitKey(0)
data = json.load(f)
outputdata = []
rectangles = []
name = param_3
img2 = cv2.imread(name, cv2.IMREAD_COLOR)
##template = cv2.imread('clutch_28_template_spring.jpeg')
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
im_substract = cv2.subtract(img1_gray, img2_gray)
im = cv2.subtract(img1, img2)
#img1 = img1_gray
#img2 = img2_gray
# plt.imshow(img2)
# plt.show()
img3 = img2.copy()
j = 1


for i in data['shapes']:

    # --------------------> IMPORTANT PARAMETER
    rectangle_size = data_jsonx["rect_size"]
    # print(i)
    part_name = i['label']
    ##########################REMOVE LATER#############################
    if part_name == 'Rivet Top 1' or part_name == "Rivet Bottom 1":
        continue
        rectangle_size = 0
    ##########################REMOVE LATER#############################

    # print(f"part_name: {part_name}")
    coords = i['points']
    l = i['points']
    crop_img = img1[int(l[0][1]):int(l[1][1])+1, int(l[0][0]):int(l[1][0])+1]
    # plt.imshow(crop_img)
    # plt.show()

    custom_th = data_jsonx["threshold"]
    # if part_name == "Rivet Top 4":
    #     custom_th = 0.83
    # elif part_name == "Rivet Bottom 2":
    #     custom_th = 0.55

    if "Rivet Top" in part_name:
        custom_th = 0.83
    elif "Rivet Bottom" in part_name:
        custom_th = 0.41  # 0.45
    elif "Torsional Spring" in part_name:
        custom_th = 0.39
        rectangle_size = 28

    b = func(img2[int(l[0][1]-rectangle_size):int(l[1][1]+rectangle_size)+1, int(l[0]
             [0]-rectangle_size):int(l[1][0]+rectangle_size)+1], crop_img, "bolt "+str(j), cus_th=custom_th)
    img_flip_ud = cv2.flip(img2[int(l[0][1]-rectangle_size):int(l[1][1]+rectangle_size)+1, int(
        l[0][0]-rectangle_size):int(l[1][0]+rectangle_size)+1], 0)  # ,crop_img,"bolt "+str(j))
    #b2 = func(img_flip_ud,crop_img,"bolt "+str(j))
    img_flip_ud = cv2.flip(img_flip_ud, 1)
    b2 = func(img_flip_ud, crop_img, "bolt "+str(j))
    '''
    if len(b2)==4 and len(b)!=4:
        b = b2
    '''
    # cv2.rectangle(img3, (int(l[0][0]), int(l[0][1])),
    #               (int(l[1][0]), int(l[1][1])), (0, 255, 0), 5)
    if len(b2) == 4:
        a, threshold, yes, cord = b2[0], b2[1], b2[2], b2[3]
        startX, startY, endX, endY = cord[0], cord[1], cord[2], cord[3]
        #cv2.rectangle(img3, (int(l[0][0])-rectangle_size+startX, int(l[0][1])-rectangle_size+startY), (int(l[0][0])-rectangle_size+endX, int(l[0][1])-rectangle_size+endY), (0, 0, 255), 5)
    # -----------------> IMPORTANT PARAMETER
    canny_threshold_lower = data_jsonx["canny_th_1"]
    # -----------------> IMPORTANT PARAMETER
    canny_threshold_upper = data_jsonx["canny_th_2"]
    rec_size = 0
    '''
    if i['label']== 'Torsional Spring 1' or i['label']== 'Torsional Spring 2' or i['label']== 'Torsional Spring 3' or i['label']== 'Torsional Spring 4':
        output = img2[int(l[0][1])+rec_size:int(l[1][1])+rec_size,int(l[0][0])-rec_size:int(l[1][0])+rec_size,:].copy()
        #plt.imshow(output)
        #plt.show()
        #output = img1[int(l[0][1])+rec_size:int(l[1][1])+rec_size,int(l[0][0])-rec_size:int(l[1][0])+rec_size,:].copy()
        #plt.imshow(output)
        #plt.show()
        #rectangle_size2 = 40
        output = img2[int(l[0][1])-rectangle_size+rec_size:int(l[1][1])+rectangle_size+rec_size,int(l[0][0])-rectangle_size-rec_size:int(l[1][0])+rectangle_size+rec_size,:].copy()
        hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([20,100,100])
        upper_blue = np.array([30,255,255])
        # Threshold the HSV image to get only blue colors
        mask_spring = cv2.inRange(hsv, lower_blue, upper_blue)
        #print(i['label'])
        ##plt.imshow(mask_spring)
        ##plt.show()
        xmin_spring = mask_spring.shape[0]
        ymin_spring = mask_spring.shape[1]
        xmax_spring = 0
        ymax_spring = 0
        yellow= 0
        for x_mask_spring in range(mask_spring.shape[0]):
            for y_mask_spring in range(mask_spring.shape[1]):
                if mask_spring[x_mask_spring][y_mask_spring] == 255:
                    yellow+=1
                    xmin_spring = min(xmin_spring,x_mask_spring)
                    ymin_spring = min(ymin_spring,y_mask_spring)
                    xmax_spring = max(xmax_spring,x_mask_spring)
                    ymax_spring = max(ymax_spring,y_mask_spring)
                #print(mask_spring[x_mask_spring][y_mask_spring])
        if yellow<=50:
            outputdata.append({ 'name': i['label'], 'present': False })
            #print('name:'+i['label'] +' present:False')#+'width: '+str(abs(startX-endX))+'height: '+str(abs(startY-endY)))
            continue
        x_test = int(l[0][0])-rectangle_size+(ymin_spring + ymax_spring)//2
        y_test = int(l[0][1])-rectangle_size+(xmin_spring + xmax_spring)//2
        x_ref = (int(l[0][0])+int(l[1][0]))//2
        y_ref = (int(l[0][1])+int(l[1][1]))//2
        x_diff = (x_test-x_ref)**2
        y_diff = (y_test-y_ref)**2
        deviation = math.sqrt(x_diff+y_diff)
        deviation = round(deviation,2)
        outputdata.append({ 'name': i['label'], 'present': True, 'dev': deviation })
        #print('name:'+i['label'] +' present:True'+' dev:'+str(deviation)) #dim:'+str(abs(startX-endX))+'x'+str(abs(startY-endY))+' dev:'+str(deviation))
        cv2.rectangle(img3, (int(l[0][0])-rectangle_size+ymin_spring, int(l[0][1])-rectangle_size+xmin_spring), (int(l[0][0])-rectangle_size+ymax_spring, int(l[0][1])-rectangle_size+xmax_spring), (0, 0, 255), 5)
        continue
    '''
    if len(b) == 3:
        a, threshold, yes = b[0], b[1], b[2]
        outputdata.append({"name": i['label'], "present": False})
        # print('name:'+i['label'] +' present:False')#+str(abs(startX-endX))+'height: '+str(abs(startY-endY)))
    else:
        # print(len(b))
        a, threshold, yes, cord = b[0], b[1], b[2], b[3]
        startX, startY, endX, endY = cord[0], cord[1], cord[2], cord[3]
        # if yes == "True":
        rec_size = 0
        cv2.rectangle(img3, (int(l[0][0])-rectangle_size+startX, int(l[0][1])-rectangle_size+startY),
                      (int(l[0][0])-rectangle_size+endX, int(l[0][1])-rectangle_size+endY), (0, 0, 255), 5)
        output = img2[int(l[0][1])-rectangle_size+startY-rec_size:int(l[0][1])-rectangle_size+endY+rec_size,
                      int(l[0][0])-rectangle_size+startX-rec_size:int(l[0][0])-rectangle_size+endX+rec_size, :].copy()
        # plt.imshow(output)
        # plt.show()
        output1 = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY).copy()
        output1 = cv2.GaussianBlur(output1, (5, 5), 0)
        output1 = cv2.Canny(output, canny_threshold_lower,
                            canny_threshold_upper)
        # plt.imshow(output1)
        # plt.show()
        x_center = output1.shape[0]//2
        y_center = output1.shape[1]//2
        sum = 0
        max_rad = 0
        min_rad = max(output1.shape[0], output1.shape[1])
        # print(math.tan(0*2*math.pi/8))
        # and (i['label']!= 'spring1' and i['label']!= 'spring2' and i['label']!= 'spring3' and i['label']!= 'spring4') :
        if i['label'] != 'Central Hub':
            x_test = int(l[0][0])-rectangle_size+(startX + endX)//2
            y_test = int(l[0][1])-rectangle_size+(startY + endY)//2
            x_ref = (int(l[0][0])+int(l[1][0]))//2
            y_ref = (int(l[0][1])+int(l[1][1]))//2
            x_diff = (x_test-x_ref)**2
            y_diff = (y_test-y_ref)**2
            deviation = math.sqrt(x_diff+y_diff)
            deviation = round(deviation, 2)
            outputdata.append(
                {"name": i['label'], "present": True, "dev": deviation})
            #print('name:'+i['label'] +' present:True'+' dev:'+str(deviation))
            #print('name:'+i['label'] +' present:'+' True dim:'+str(abs(startX-endX))+'x'+str(abs(startY-endY))+' dev:'+str(deviation))
            continue
        rec_size = 0
        output = img2[int(l[0][1])-rectangle_size+startY-rec_size:int(l[0][1])-rectangle_size+endY+rec_size,
                      int(l[0][0])-rectangle_size+startX-rec_size:int(l[0][0])-rectangle_size+endX+rec_size, :].copy()
        # plt.imshow(output)
        # plt.show()
        output1 = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY).copy()
        output1 = cv2.GaussianBlur(output1, (5, 5), 0)
        output1 = cv2.Canny(output, canny_threshold_lower,
                            canny_threshold_upper)
        # plt.imshow(output1)
        # plt.show()
        # ------------------------------------< IMPORANT PARAMETER
        step = data_jsonx["step"]
        count = 0
        for i in range(0, int(step/2)):
            d = 0
            x_edge = x_center
            y_edge = y_center
            if i/step == 0.25:
                count += 1
                continue
            for j in range(x_center, output.shape[0]):
                # d+=1
                x = j
                y = int(math.tan(i*2*math.pi/step)*(x-x_center)+y_center)
                if y >= output.shape[1]:
                    x_edge = x
                    y_edge = y
                    break
                '''
                if y<0:
                    x_edge = x
                    y_edge = y
                    break
                '''
                # print(output1[x][y])
                output[x][y] = [0, 0, 255]
                try:
                    if output1[x][y] != 0 or output1[x-1][y] != 0 or output1[x+1][y] != 0 or output1[x][y-1] != 0 or output1[x][y+1] != 0 or output1[x-1][y-1] != 0 or output1[x-1][y+1] != 0 or output1[x+1][y-1] != 0 or output1[x+1][y+1] != 0:
                        x_edge = x
                        y_edge = y
                        break
                except:
                    x_edge = x
                    y_edge = y
                    break
            # plt.imshow(output)
            # plt.show()
            # print(math.sqrt((x_edge-x_center)**2+(y_edge-y_center)**2))
            le = math.sqrt((x_edge-x_center)**2+(y_edge-y_center)**2)
            max_rad = max(le, max_rad)
            min_rad = min(le, min_rad)
            sum += le
        for i in range(int(step/2), step):
            d = 0
            x_edge = x_center
            y_edge = y_center
            if i/step == 0.75:
                count += 1
                continue
            for j in range(x_center-1, -1, -1):
                # d+=1
                x = j
                y = int(math.tan(i*2*math.pi/step)*(x-x_center)+y_center)
                if y >= output.shape[1]:
                    x_edge = x
                    y_edge = output.shape[1]
                    break
                if y < 0:
                    x_edge = x
                    y_edge = y
                    break
                # print(output1[x][y])
                output[x][y] = [0, 0, 255]
                try:
                    if output1[x][y] != 0 or output1[x-1][y] != 0 or output1[x+1][y] != 0 or output1[x][y-1] != 0 or output1[x][y+1] != 0 or output1[x-1][y-1] != 0 or output1[x-1][y+1] != 0 or output1[x+1][y-1] != 0 or output1[x+1][y+1] != 0:
                        x_edge = x
                        y_edge = y
                        break
                except:
                    x_edge = x
                    y_edge = y
                    break
            # plt.imshow(output)
            # plt.show()
            # print(math.sqrt((x_edge-x_center)**2+(y_edge-y_center)**2),x_center,y_edge,y_center,i,math.tan(i*2*math.pi/step),i*2*math.pi/step)
            le = math.sqrt((x_edge-x_center)**2+(y_edge-y_center)**2)
            max_rad = max(max_rad, le)
            min_rad = min(le, min_rad)
            sum += le
            # plt.imshow(output)
            # plt.show()
        # ------------------------- IMPORTANT PARAMETER
        avg_rad = int((max_rad+min_rad)//2) + data_jsonx["avg_rad_extra"]
        #avg_rad = int(min_rad)
        #print("average radius in pixels: "+str(avg_rad))
        output[x_center][y_center] = [255, 0, 0]
        # output[x_center][y_center+avg_rad]=[255,0,0]
        points = []
        lis = []
        quad1 = set()
        quad2 = set()
        quad3 = set()
        quad4 = set()
        for j in range(x_center, x_center+int(avg_rad)+1):
            o = int(math.sqrt(avg_rad**2-(j-x_center)**2))
            y1 = y_center+o
            y2 = y_center-o
            # print(y1,y2)
            output[j][y1] = [255, 0, 0]
            output[j][y2] = [255, 0, 0]
            points.append([j, y1])
            quad1.add((j, y1))
            points.append([j, y2])
            quad2.add((j, y2))
        for j in range(x_center, x_center-int(avg_rad)-1, -1):
            o = int(math.sqrt(avg_rad**2-(j-x_center)**2))
            y1 = y_center+o
            y2 = y_center-o
            # print(y1,y2)
            output[j][y1] = [255, 0, 0]
            output[j][y2] = [255, 0, 0]
            points.append([j, y1])
            quad3.add((j, y2))
            points.append([j, y2])
            quad4.add((j, y1))
        for j in range(y_center, y_center+int(avg_rad)+1):
            o = int(math.sqrt(avg_rad**2-(j-y_center)**2))
            x1 = x_center+o
            x2 = x_center-o
            # print(x1,x2)
            output[x1][j] = [255, 0, 0]
            output[x2][j] = [255, 0, 0]
            quad1.add((x1, j))
            points.append([x1, j])
            quad4.add((x2, j))
            points.append([x2, j])
        for j in range(y_center, y_center-int(avg_rad)-1, -1):
            o = int(math.sqrt(avg_rad**2-(j-y_center)**2))
            x1 = x_center+o
            x2 = x_center-o
            # print(x1,x2)
            output[x1][j] = [255, 0, 0]
            output[x2][j] = [255, 0, 0]
            quad2.add((x1, j))
            points.append([x1, j])
            quad3.add((x2, j))
            points.append([x2, j])
        # print(step-count)
        radius = sum/(step-count)
        #print("average radius in pixels: "+str(radius))
        # print(radius)
        # plt.imshow(output)
        # plt.show()
        # circle detection
        output = img2[int(l[0][1])-rectangle_size+startY-rec_size:int(l[0][1])-rectangle_size+endY+rec_size,
                      int(l[0][0])-rectangle_size+startX-rec_size:int(l[0][0])-rectangle_size+endX+rec_size, :].copy()
        #output = img2[int(l[0][1])-rectangle_size+startY:int(l[0][1])-rectangle_size+endY,int(l[0][0])-rectangle_size+startX:int(l[0][0])-rectangle_size+endX,:].copy()
        out_blur = cv2.GaussianBlur(output, (5, 5), 0).copy()
        output1 = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY).copy()
        output1 = cv2.GaussianBlur(output1, (5, 5), 0)
        #out_blur = output1.copy()
        output1 = cv2.Canny(output1, canny_threshold_lower,
                            canny_threshold_upper)
        output2 = cv2.bitwise_not(output1).copy()
        # cv2.imwrite('b.jpg',output2)
        #output2 = cv2.imread('b.jpg')
        # cv2.imwrite('a.jpg', output1)
        # output1 = cv2.imread('a.jpg')
        # plt.imshow(output1)
        # plt.show()
        #output2 = cv2.cvtColor(output2, cv2.COLOR_BGR2GRAY).copy()
        # plt.imshow(output2)
        # plt.show()
        '''
        circles = cv2.HoughCircles(output2, cv2.HOUGH_GRADIENT, 1.2, 1)

        # ensure at least some circles were found
		if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circle
            for (x, y, r) in circles:
                cv2.circle(output, (x, y), r, (255, 0, 0), 4)
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        if circles is not None:
            len(circles)
        plt.imshow(output)
        plt.show()
        '''
        kernel = np.ones((3, 3), np.uint8)
        # cv2.imwrite('b.jpg',output2)
        #output2 = cv2.imread('b.jpg')
        mask = output2.copy()
        # -----------------> IMPORTANT PARAMETER
        mask = cv2.erode(output2, kernel, iterations=data_jsonx["erode"])
        #mask = cv2.dilate(mask, kernel, iterations=1)
        #mask = output2.copy()
        # plt.imshow(mask)
        # plt.show()
        #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        #closing = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = mask.astype('uint8')
        contours, _ = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #contours,_ = cv2.findContours(closing, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #contours,_ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #print("finished finding Counters")
        # Create the marker image for the watershed algorithm
        markers = np.zeros(mask.shape, dtype=np.int32)
        # Draw the foreground markers
        # print(len(contours))
        length_of_counters = len(contours)
        for k in range(length_of_counters):
            # print((k/length_of_counters)*100,end="\r")
            cv2.drawContours(markers, contours, k, (k+1), -1)
        cv2.circle(markers, (5, 5), 3, (255, 255, 255), -1)
        # cv2.imwrite("markers_02_back.jpg", markers)
        output = img2[int(l[0][1])-rectangle_size+startY-rec_size:int(l[0][1])-rectangle_size+endY+rec_size,
                      int(l[0][0])-rectangle_size+startX-rec_size:int(l[0][0])-rectangle_size+endX+rec_size, :].copy()
        #output = img2[int(l[0][1])-rectangle_size+startY:int(l[0][1])-rectangle_size+endY,int(l[0][0])-rectangle_size+startX:int(l[0][0])-rectangle_size+endX,:].copy()
        instances = cv2.watershed(output, markers)
        #print("instances ",instances.shape)
        # plt.imshow(markers)
        # plt.show()
        colors = []
        k = 0
        for contour in contours:
            # print((k/length_of_counters)*100,end="\r")
            c = (random.randint(0, 256), random.randint(
                0, 256), random.randint(0, 256))
            while c == (255, 0, 0):
                c = (random.randint(0, 256), random.randint(
                    0, 256), random.randint(0, 256))
            colors.append(c)
            k = k+1
        # Create the result image
        dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
        o = 0
        for k in range(markers.shape[0]):
            for l in range(markers.shape[1]):
                # print((o/(markers.shape[0]*markers.shape[1]))*100,end="\r")
                index = markers[k, l]
                if index > 0 and index <= len(contours):
                    dst[k, l, :] = colors[index-1]
                o = o+1
        # plt.imshow(dst)
        # plt.show()
        points = []
        quad1 = list(quad1)
        quad1.sort(key=letter_cmp_key)
        quad2 = list(quad2)
        quad2.sort(reverse=True)
        quad3 = list(quad3)
        quad3.sort(key=letter_cmp_key2, reverse=True)
        quad4 = list(quad4)
        quad4.sort()
        points = quad1+quad2+quad3+quad4
        start = None
        count = 0
        dst2 = dst.copy()
        mid = dst[output.shape[0]//2][output.shape[1]//2]
        # print(mid)
        le = len(points)
        # for k in range(0,le):
        k = 0
        while k < le:
            # dst[k[0]][k[1]]=[255,0,0]
            # print(dst[points[k][0]][points[k][1]])#,type(dst[k[0]][k[1]]),dst[k[0]][k[1]][0],dst[k[0]][k[1]][1],dst[k[0]][k[1]][2])
            h = dst[points[k][0]][points[k][1]]
            '''
            if int(start[0]) != 0 and int(start[1]) != 0 and int(start[2]) != 0 and int(h[0]) == 0 and int(h[1]) == 0 and int(h[2]) == 0:
                count+=1
                dst2[k[0]][k[1]]=[255,0,0]
                start = h
            else:
                dst2[k[0]][k[1]]=[0,0,255]
                start = h
                continue
            '''
            if int(h[0]) == 0 and int(h[1]) == 0 and int(h[2]) == 0:
                # dst2[k[0]][k[1]]=[255,0,0]
                k += 1
                continue
            if start is None:
                start = h
                k += 1
                continue
            if int(h[0]) == start[0] and int(h[1]) == start[1] and int(h[2]) == start[2]:
                dst2[points[k][0]][points[k][1]] = [0, 0, 255]
                k += 1
                continue
            else:
                count += 1
                dst2[points[k][0]][points[k][1]] = [255, 0, 0]
                if le-k < 20:
                    k += 1
                    start = h
                else:
                    k += 8
                    start = None
                # print(start)
                #start = None
                #start = h
                # dst2[points[k][0]][points[k][1]]=[255,0,0]
            # cv2.imshow('a',dst)
            # plt.imshow(dst)
            # plt.show()
            '''
            if start is None:
                start = h
                continue
            if int(h[0]) == mid[0] and int(h[1]) == mid[1] and int(h[2]) == mid[2] and (int(start[0]) != mid[0] and int(start[1]) != mid[1] and int(start[2]) != mid[2]):
                count+=1
                dst2[k[0]][k[1]]=[0,0,255]
                start = h
            else:
                dst2[k[0]][k[1]]=[255,0,0]
                start = h
            '''
        count1 = count
        count = 0
        dst2 = dst.copy()
        mid = dst[output.shape[0]//2][output.shape[1]//2]
        # print(mid)
        le = len(points)
        # for k in range(0,le):
        k = 0
        points = points[5:]+points[:5:-1]
        while k < le:
            # dst[k[0]][k[1]]=[255,0,0]
            # print(dst[points[k][0]][points[k][1]])#,type(dst[k[0]][k[1]]),dst[k[0]][k[1]][0],dst[k[0]][k[1]][1],dst[k[0]][k[1]][2])
            h = dst[points[k][0]][points[k][1]]
            '''
            if int(start[0]) != 0 and int(start[1]) != 0 and int(start[2]) != 0 and int(h[0]) == 0 and int(h[1]) == 0 and int(h[2]) == 0:
                count+=1
                dst2[k[0]][k[1]]=[255,0,0]
                start = h
            else:
                dst2[k[0]][k[1]]=[0,0,255]
                start = h
                continue
            '''
            if int(h[0]) == 0 and int(h[1]) == 0 and int(h[2]) == 0:
                # dst2[k[0]][k[1]]=[255,0,0]
                k += 1
                continue
            if start is None:
                start = h
                k += 1
                continue
            if int(h[0]) == start[0] and int(h[1]) == start[1] and int(h[2]) == start[2]:
                dst2[points[k][0]][points[k][1]] = [0, 0, 255]
                k += 1
                continue
            else:
                count += 1
                dst2[points[k][0]][points[k][1]] = [255, 0, 0]
                if le-k < 20:
                    k += 1
                    start = h
                else:
                    # -----------------------> IMPORTANT PARAMETER
                    k += data_jsonx["k"]
                    start = None
                # print(start)
                #start = None
                #start = h
                # dst2[points[k][0]][points[k][1]]=[255,0,0]
            # cv2.imshow('a',dst)
            # plt.imshow(dst)
            # plt.show()
            '''
            if start is None:
                start = h
                continue
            if int(h[0]) == mid[0] and int(h[1]) == mid[1] and int(h[2]) == mid[2] and (int(start[0]) != mid[0] and int(start[1]) != mid[1] and int(start[2]) != mid[2]):
                count+=1
                dst2[k[0]][k[1]]=[0,0,255]
                start = h
            else:
                dst2[k[0]][k[1]]=[255,0,0]
                start = h
            '''
        count2 = count
        count = max(count1, count2)
        dst2[points[-1][0]][points[-1][1]] = [0, 255, 0]
        # print(mid)

        # print("no of ridges :"+str(count//2))#,len(points))
        # plt.imshow(dst)
        # plt.show()
        # cv2.imwrite('counters.jpg', dst2)
        # plt.imshow(dst2)
        # plt.show()
        # cv2.imwrite(name.split('.')[0]+"circle.jpg", dst2)
        '''
        contours.sort(key=lambda x:cv2.boundingRect(x)[0])
        array = []
        ii = 1
        print(len(contours))
        for c in contours:
            (x,y),r = cv2.minEnclosingCircle(c)
            center = (int(x),int(y))
            r = int(r)
            #if r >= 6 and r<=10:
            #print(r)
            cv2.circle(output,center,r,(0,255,0),1)
            cv2.rectangle(output, (center[0]-1, center[1] - 1), (center[0] + 1, center[1] + 1), (0, 128, 255), -1)
            array.append(center)
        ##plt.imshow(output)
        ##plt.show()
        ##cv2.putText(img3, str(threshold), (int(l[0][0]), int(l[0][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
        '''
        mid = dst[output.shape[0]//2][output.shape[1]//2]
        # print(mid)
        xmin_center = dst.shape[0]
        ymin_center = dst.shape[1]
        xmax_center = 0
        ymax_center = 0
        for x_dst in range(dst.shape[0]):
            for y_dst in range(dst.shape[1]):
                if dst[x_dst][y_dst][0] == mid[0] and dst[x_dst][y_dst][1] == mid[1] and dst[x_dst][y_dst][2] == mid[2]:
                    xmin_center = min(xmin_center, x_dst)
                    ymin_center = min(ymin_center, y_dst)
                    xmax_center = max(xmax_center, x_dst)
                    ymax_center = max(ymax_center, y_dst)
        #cv2.rectangle(dst, (ymin_center, xmin_center), (ymax_center, xmax_center), (0, 0, 255), 5)
        # plt.imshow(dst)
        # plt.show()
        output = img2[int(coords[0][1])-rectangle_size+startY-rec_size:int(coords[0][1])-rectangle_size+endY+rec_size,
                      int(coords[0][0])-rectangle_size+startX-rec_size:int(coords[0][0])-rectangle_size+endX+rec_size, :].copy()
        x_center = (xmin_center+xmax_center)//2
        y_center = (ymin_center+ymax_center)//2
        # plt.imshow(output)
        # plt.show()
        output1 = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY).copy()
        output1 = cv2.GaussianBlur(output1, (5, 5), 0)
        output1 = cv2.Canny(output, canny_threshold_lower,
                            canny_threshold_upper)
        # plt.imshow(output1)
        # plt.show()
        max_rad = 0
        min_rad = max(output1.shape[0], output1.shape[1])
        step = 10
        count = 0
        sum = 0
        for i in range(0, int(step/2)):
            d = 0
            x_edge = x_center
            y_edge = y_center
            if i/step == 0.25:
                count += 1
                continue
            for j in range(x_center, output.shape[0]):
                # d+=1
                x = j
                y = int(math.tan(i*2*math.pi/step)*(x-x_center)+y_center)
                if y >= output.shape[1]:
                    x_edge = x
                    y_edge = y  # output.shape[1]-1
                    break
                '''
                if y<0:
                    x_edge = x
                    y_edge = y
                    break
                '''
                # print(output1[x][y])
                output[x][y] = [0, 0, 255]
                try:
                    if output1[x][y] != 0 or output1[x-1][y] != 0 or output1[x+1][y] != 0 or output1[x][y-1] != 0 or output1[x][y+1] != 0 or output1[x-1][y-1] != 0 or output1[x-1][y+1] != 0 or output1[x+1][y-1] != 0 or output1[x+1][y+1] != 0:
                        x_edge = x
                        y_edge = y
                        break
                except:
                    x_edge = x
                    y_edge = y
                    break
            # plt.imshow(output)
            # plt.show()
            # print(math.sqrt((x_edge-x_center)**2+(y_edge-y_center)**2))
            le = math.sqrt((x_edge-x_center)**2+(y_edge-y_center)**2)
            max_rad = max(le, max_rad)
            min_rad = min(le, min_rad)
            sum += le
        for i in range(int(step/2), step):
            d = 0
            x_edge = x_center
            y_edge = y_center
            if i/step == 0.75:
                count += 1
                continue
            for j in range(x_center-1, -1, -1):
                # d+=1
                x = j
                y = int(math.tan(i*2*math.pi/step)*(x-x_center)+y_center)
                if y >= output.shape[1]:
                    x_edge = x
                    y_edge = y  # output.shape[1]-1
                    break
                if y < 0:
                    x_edge = x
                    y_edge = y
                    break
                # print(output1[x][y])
                output[x][y] = [0, 0, 255]
                try:
                    if output1[x][y] != 0 or output1[x-1][y] != 0 or output1[x+1][y] != 0 or output1[x][y-1] != 0 or output1[x][y+1] != 0 or output1[x-1][y-1] != 0 or output1[x-1][y+1] != 0 or output1[x+1][y-1] != 0 or output1[x+1][y+1] != 0:
                        x_edge = x
                        y_edge = y
                        break
                except:
                    x_edge = x
                    y_edge = y
                    break
            # plt.imshow(output)
            # plt.show()
            # print(math.sqrt((x_edge-x_center)**2+(y_edge-y_center)**2),x_center,y_edge,y_center,i,math.tan(i*2*math.pi/step),i*2*math.pi/step)
            le = math.sqrt((x_edge-x_center)**2+(y_edge-y_center)**2)
            max_rad = max(max_rad, le)
            min_rad = min(le, min_rad)
            sum += le
        # plt.imshow(output)
        # plt.show()
        avg_rad = int((max_rad+min_rad)//2)+2
        #avg_rad = int(min_rad)
        #print("average radius in pixels: "+str(avg_rad))
        output[x_center][y_center] = [255, 0, 0]
        # output[x_center][y_center+avg_rad]=[255,0,0]
        points = []
        lis = []
        quad1 = set()
        quad2 = set()
        quad3 = set()
        quad4 = set()
        for j in range(x_center, x_center+int(avg_rad)+1):
            o = int(math.sqrt(avg_rad**2-(j-x_center)**2))
            y1 = y_center+o
            y2 = y_center-o
            # print(y1,y2)
            output[j][y1] = [255, 0, 0]
            output[j][y2] = [255, 0, 0]
            points.append([j, y1])
            quad1.add((j, y1))
            points.append([j, y2])
            quad2.add((j, y2))
        for j in range(x_center, x_center-int(avg_rad)-1, -1):
            o = int(math.sqrt(avg_rad**2-(j-x_center)**2))
            y1 = y_center+o
            y2 = y_center-o
            # print(y1,y2)
            output[j][y1] = [255, 0, 0]
            output[j][y2] = [255, 0, 0]
            points.append([j, y1])
            quad3.add((j, y2))
            points.append([j, y2])
            quad4.add((j, y1))
        for j in range(y_center, y_center+int(avg_rad)+1):
            o = int(math.sqrt(avg_rad**2-(j-y_center)**2))
            x1 = x_center+o
            x2 = x_center-o
            # print(x1,x2)
            output[x1][j] = [255, 0, 0]
            output[x2][j] = [255, 0, 0]
            quad1.add((x1, j))
            points.append([x1, j])
            quad4.add((x2, j))
            points.append([x2, j])
        for j in range(y_center, y_center-int(avg_rad)-1, -1):
            o = int(math.sqrt(avg_rad**2-(j-y_center)**2))
            x1 = x_center+o
            x2 = x_center-o
            # print(x1,x2)
            output[x1][j] = [255, 0, 0]
            output[x2][j] = [255, 0, 0]
            quad2.add((x1, j))
            points.append([x1, j])
            quad3.add((x2, j))
            points.append([x2, j])
        # print(step-count)
        radius = sum/(step-count)
        #print("average radius in pixels: "+str(radius))
        outputdata.append({"name": "Central Hub", "present": True})
        #print('name:Central Hub'+' present:True ',end = '')
        # print(radius)
        # plt.imshow(output)
        # plt.show()
        # circle detection
        output = img2[int(coords[0][1])-rectangle_size+startY-rec_size:int(coords[0][1])-rectangle_size+endY+rec_size,
                      int(coords[0][0])-rectangle_size+startX-rec_size:int(coords[0][0])-rectangle_size+endX+rec_size, :].copy()
        #output = img2[int(l[0][1])-rectangle_size+startY:int(l[0][1])-rectangle_size+endY,int(l[0][0])-rectangle_size+startX:int(l[0][0])-rectangle_size+endX,:].copy()
        out_blur = cv2.GaussianBlur(output, (5, 5), 0).copy()
        output1 = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY).copy()
        output1 = cv2.GaussianBlur(output1, (5, 5), 0)
        #out_blur = output1.copy()
        output1 = cv2.Canny(output1, canny_threshold_lower,
                            canny_threshold_upper)
        output2 = cv2.bitwise_not(output1).copy()
        # cv2.imwrite('b.jpg',output2)
        #output2 = cv2.imread('b.jpg')
        # cv2.imwrite('a.jpg', output1)
        # output1 = cv2.imread('a.jpg')
        # plt.imshow(output1)
        # plt.show()
        #output2 = cv2.cvtColor(output2, cv2.COLOR_BGR2GRAY).copy()
        # plt.imshow(output2)
        # plt.show()
        '''
        circles = cv2.HoughCircles(output2, cv2.HOUGH_GRADIENT, 1.2, 1)

        # ensure at least some circles were found
		if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circle
            for (x, y, r) in circles:
                cv2.circle(output, (x, y), r, (255, 0, 0), 4)
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        if circles is not None:
            len(circles)
        plt.imshow(output)
        plt.show()
        '''
        kernel = np.ones((3, 3), np.uint8)
        # cv2.imwrite('b.jpg',output2)
        #output2 = cv2.imread('b.jpg')
        mask = output2.copy()
        # -----------------------< IMPORANT PARAMETER
        mask = cv2.erode(output2, kernel, iterations=data_jsonx["erode"])
        #mask = cv2.dilate(mask, kernel, iterations=1)
        #mask = output2.copy()
        # plt.imshow(mask)
        # plt.show()
        #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        #closing = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = mask.astype('uint8')
        contours, _ = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #contours,_ = cv2.findContours(closing, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #contours,_ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #print("finished finding Counters")
        # Create the marker image for the watershed algorithm
        markers = np.zeros(mask.shape, dtype=np.int32)
        # Draw the foreground markers
        # print(len(contours))
        length_of_counters = len(contours)
        for k in range(length_of_counters):
            # print((k/length_of_counters)*100,end="\r")
            cv2.drawContours(markers, contours, k, (k+1), -1)
        cv2.circle(markers, (5, 5), 3, (255, 255, 255), -1)
        # cv2.imwrite("markers_02_back.jpg", markers)
        output = img2[int(coords[0][1])-rectangle_size+startY-rec_size:int(coords[0][1])-rectangle_size+endY+rec_size,
                      int(coords[0][0])-rectangle_size+startX-rec_size:int(coords[0][0])-rectangle_size+endX+rec_size, :].copy()
        #output = img2[int(l[0][1])-rectangle_size+startY:int(l[0][1])-rectangle_size+endY,int(l[0][0])-rectangle_size+startX:int(l[0][0])-rectangle_size+endX,:].copy()
        instances = cv2.watershed(output, markers)
        #print("instances ",instances.shape)
        # plt.imshow(markers)
        # plt.show()
        colors = []
        k = 0
        for contour in contours:
            # print((k/length_of_counters)*100,end="\r")
            c = (random.randint(0, 256), random.randint(
                0, 256), random.randint(0, 256))
            while c == (255, 0, 0):
                c = (random.randint(0, 256), random.randint(
                    0, 256), random.randint(0, 256))
            colors.append(c)
            k = k+1
        # Create the result image
        dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
        o = 0
        for k in range(markers.shape[0]):
            for l in range(markers.shape[1]):
                # print((o/(markers.shape[0]*markers.shape[1]))*100,end="\r")
                index = markers[k, l]
                if index > 0 and index <= len(contours):
                    dst[k, l, :] = colors[index-1]
                o = o+1
        # plt.imshow(dst)
        # plt.show()
        points = []
        quad1 = list(quad1)
        quad1.sort(key=letter_cmp_key)
        quad2 = list(quad2)
        quad2.sort(reverse=True)
        quad3 = list(quad3)
        quad3.sort(key=letter_cmp_key2, reverse=True)
        quad4 = list(quad4)
        quad4.sort()
        points = quad1+quad2+quad3+quad4
        start = None
        count = 0
        dst2 = dst.copy()
        mid = dst[output.shape[0]//2][output.shape[1]//2]
        # print(mid)
        le = len(points)
        # for k in range(0,le):
        k = 0
        while k < le:
            # dst[k[0]][k[1]]=[255,0,0]
            # print(dst[points[k][0]][points[k][1]])#,type(dst[k[0]][k[1]]),dst[k[0]][k[1]][0],dst[k[0]][k[1]][1],dst[k[0]][k[1]][2])
            h = dst[points[k][0]][points[k][1]]
            '''
            if int(start[0]) != 0 and int(start[1]) != 0 and int(start[2]) != 0 and int(h[0]) == 0 and int(h[1]) == 0 and int(h[2]) == 0:
                count+=1
                dst2[k[0]][k[1]]=[255,0,0]
                start = h
            else:
                dst2[k[0]][k[1]]=[0,0,255]
                start = h
                continue
            '''
            if int(h[0]) == 0 and int(h[1]) == 0 and int(h[2]) == 0:
                # dst2[k[0]][k[1]]=[255,0,0]
                k += 1
                continue
            if start is None:
                start = h
                k += 1
                continue
            if int(h[0]) == start[0] and int(h[1]) == start[1] and int(h[2]) == start[2]:
                dst2[points[k][0]][points[k][1]] = [0, 0, 255]
                k += 1
                continue
            else:
                count += 1
                dst2[points[k][0]][points[k][1]] = [255, 0, 0]
                if le-k < 20:
                    k += 1
                    start = h
                else:
                    # ------------------------------< IMPORANT PARAMETER
                    k += data_jsonx["k"]
                    start = None
                # print(start)
                #start = None
                #start = h
                # dst2[points[k][0]][points[k][1]]=[255,0,0]
            # cv2.imshow('a',dst)
            # plt.imshow(dst)
            # plt.show()
            '''
            if start is None:
                start = h
                continue
            if int(h[0]) == mid[0] and int(h[1]) == mid[1] and int(h[2]) == mid[2] and (int(start[0]) != mid[0] and int(start[1]) != mid[1] and int(start[2]) != mid[2]):
                count+=1
                dst2[k[0]][k[1]]=[0,0,255]
                start = h
            else:
                dst2[k[0]][k[1]]=[255,0,0]
                start = h
            '''
        count1 = count
        count = 0
        dst2 = dst.copy()
        mid = dst[output.shape[0]//2][output.shape[1]//2]
        # print(mid)
        le = len(points)
        # for k in range(0,le):
        k = 0
        points = points[5:]+points[:5:-1]
        while k < le:
            # dst[k[0]][k[1]]=[255,0,0]
            # print(dst[points[k][0]][points[k][1]])#,type(dst[k[0]][k[1]]),dst[k[0]][k[1]][0],dst[k[0]][k[1]][1],dst[k[0]][k[1]][2])
            h = dst[points[k][0]][points[k][1]]
            '''
            if int(start[0]) != 0 and int(start[1]) != 0 and int(start[2]) != 0 and int(h[0]) == 0 and int(h[1]) == 0 and int(h[2]) == 0:
                count+=1
                dst2[k[0]][k[1]]=[255,0,0]
                start = h
            else:
                dst2[k[0]][k[1]]=[0,0,255]
                start = h
                continue
            '''
            if int(h[0]) == 0 and int(h[1]) == 0 and int(h[2]) == 0:
                # dst2[k[0]][k[1]]=[255,0,0]
                k += 1
                continue
            if start is None:
                start = h
                k += 1
                continue
            if int(h[0]) == start[0] and int(h[1]) == start[1] and int(h[2]) == start[2]:
                dst2[points[k][0]][points[k][1]] = [0, 0, 255]
                k += 1
                continue
            else:
                count += 1
                dst2[points[k][0]][points[k][1]] = [255, 0, 0]
                if le-k < 20:
                    k += 1
                    start = h
                else:
                    k += 8
                    start = None
                # print(start)
                #start = None
                #start = h
                # dst2[points[k][0]][points[k][1]]=[255,0,0]
            # cv2.imshow('a',dst)
            # plt.imshow(dst)
            # plt.show()
            '''
            if start is None:
                start = h
                continue
            if int(h[0]) == mid[0] and int(h[1]) == mid[1] and int(h[2]) == mid[2] and (int(start[0]) != mid[0] and int(start[1]) != mid[1] and int(start[2]) != mid[2]):
                count+=1
                dst2[k[0]][k[1]]=[0,0,255]
                start = h
            else:
                dst2[k[0]][k[1]]=[255,0,0]
                start = h
            '''
        count2 = count
        count = max(count1, count2)
        dst2[points[-1][0]][points[-1][1]] = [0, 255, 0]
        # print(mid)
        x_test = int(coords[0][0])-rectangle_size-rec_size+(startX+endX)//2
        y_test = int(coords[0][1])-rectangle_size-rec_size+(startY+endY)//2
        x_ref = (int(coords[0][0])+int(coords[1][0]))//2
        y_ref = (int(coords[0][1])+int(coords[1][1]))//2
        x_diff = (x_test-x_ref)**2
        y_diff = (y_test-y_ref)**2
        deviation = math.sqrt(x_diff+y_diff)
        deviation = round(deviation, 2)
        outputdata[-1]["dim"] = round(radius, 2)
        outputdata[-1]["dev"] = deviation
        # print("dim :"+str(round(radius,2))+'/'+str(count//2)+' dev:'+str(deviation))#,len(points))
        #print(i['label'] +' is present '+'radius: '+str(radius)+'height: ',end = '\r')
        # plt.imshow(dst)
        # plt.show()
        # cv2.imwrite('counters.jpg', dst2)
        # plt.imshow(dst2)
        # plt.show()
        # cv2.imwrite(name.split('.')[0]+"circle.jpg", dst2)
        '''
        contours.sort(key=lambda x:cv2.boundingRect(x)[0])
        array = []
        ii = 1
        print(len(contours))
        for c in contours:
            (x,y),r = cv2.minEnclosingCircle(c)
            center = (int(x),int(y))
            r = int(r)
            #if r >= 6 and r<=10:
            #print(r)
            cv2.circle(output,center,r,(0,255,0),1)
            cv2.rectangle(output, (center[0]-1, center[1] - 1), (center[0] + 1, center[1] + 1), (0, 128, 255), -1)
            array.append(center)
        ##plt.imshow(output)
        ##plt.show()
        ##cv2.putText(img3, str(threshold), (int(l[0][0]), int(l[0][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
        '''

    j = j+1
# plt.imshow(img3)
# plt.show()
final_outx = []


stop_pinx = ["Central Hub", "Stop Pin 1", "Stop Pin 2",
             "Stop Pin 3", "Stop Pin 4", "Stop Pin 5", "Stop Pin 6"]


for s in stop_pinx:
    pin = [j for j in outputdata if j["name"] == s]

    final_outx.append(pin[0])


not_finalx = [i for i in outputdata if i["name"] not in stop_pinx]


final_outx.extend(not_finalx)
print(json.dumps(final_outx, indent=2))
# print(outputdata)
# cv2.imwrite(name.split('.')[0]+'_'+'result.jpg',img3)

cv2.imwrite(f'{BASE_PATH}/images/result.jpg', img3)
