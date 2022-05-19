# For notebook type cell creation #%%
import sys
import cv2
import numpy as np
import math
import os
import time
import json

BASE_PATH = "/home/user/frinks/python_backend"

# --------------------- MAIN FUNCTION --------------------


def detect_circle(img, rmin=15, rmax=30, adp_th=103, canny_th1=0, canny_th2=255, circle_pos=1):

    # ------------------------------ PARAMETER CONTROL ------------------------------

    # exp = "single_crop"  # AdoptiveThreshold103_Canny_diate
    # save = True
    imsz = "full"
    rmin = rmin
    rmax = rmax
    rad_list = []

    # save_dirx = f"alt_exp/{exp}/"
    # os.makedirs(save_dirx, exist_ok=True)

    # print(f"---------- Working with img: {img_name} ----------")

    # ----------------------------- READING IMAGE ----------------------------------------------------------------
    # 3 channel image to plot the output

    start_time = time.time()
    img_c = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # cv2.imread(img_path,1)

    ad_th = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, adp_th, 3)
    imgp = ad_th.copy()

    canny = cv2.Canny(imgp, canny_th1, canny_th2)

    # Kernel for dilating
    kernel = np.ones((2, 2), dtype=np.int8)
    # Dilating the output of canny
    # This  makes the edges more defined
    canny = cv2.dilate(canny, kernel, iterations=1)
    # Finding the contours from the image
    contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Getting a canvas to plot the output
    canvas = img_c.copy()
    # Drawing the contours
    # cv2.drawContours(canvas, contours, -1, (255, 0, 0), 1)
    # Filtering the contours
    contours = [contour for contour in contours if len(contour) > 15]
    # Getting the minimum enclosing circles
    circles = [cv2.minEnclosingCircle(contour) for contour in contours]
    # Getting the area of each contours
    areas = [cv2.contourArea(contour) for contour in contours]
    # Calculating the radius fro the contour area
    radiuses = [math.sqrt(area / math.pi) for area in areas]

    # print("----XXXX after radius creation ---")

    # variable for the center count
    center_no = 0
    # for storing the circle detection
    plots = []
    px_prev = []

    # print("------------- LOOPING through detected circles !!!!")
    for circle, radius in zip(circles, radiuses):
        nearby = False
        if (int(radius) != 0):
            if 0.85 <= (circle[1] / radius) <= 1.15:
                p = (round(circle[0][0]), round(circle[0][1]))
                r = round(circle[1])

    # print("--------------------------****-------------------------")

    # print(f"p_centroid_coordinate:{p}")
    # print(f"radius: {r} px")
    # print(f"center_no:{center_no}")
                if imsz == 'full':

                    if rmin < r < rmax and (p[0]not in [cc[0] for cc in px_prev]):

                        for nc in px_prev:
                            # print("------------ inside nc abs check")
                            cx1 = nc[0]
                            cy1 = nc[1]

                            cx2 = p[0]
                            cy2 = p[1]

                            # Calculating euclidean distance between two circle centroids

                            c_dis = int(
                                math.sqrt(((cx2 - cx1) ** 2) + ((cy2 - cy1) ** 2)) ** 0.5)
                            # print(
                            #     f"center:{center_no}--->{(nc,p)} ---- diff: {c_dis}")
                            if c_dis <= 3:
                                nearby = True
                                # print(f"nearby inside if check: {nearby}")
                                # print(
                                #     f"------------ this centroid is closer to nc: {nc}")
                                break

                        # print(f"nearby: {nearby}")
                        # if nearby == True:

                            # print(
                            #     f"Duplicate circle detected !!! Hence skipping this circle with centroid:{p}")
                        else:
                            plots.append([p, r, center_no, radius])
                            # cv2.circle(canvas, p, r, (0, 255, 0), 1)
                            # cv2.putText(canvas, f"{p}", p, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255),1)
                            # cv2.putText(
                            #     canvas, f"{r}", p, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                            # cv2.putText(canvas, f"{center_no}--> {p}", (p[0]+r+5, p[1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255),1)

                            center_no += 1
                            px_prev.append(p)
                            # cv2.imwrite(f"{save_dirx}/"+f"cp_img_{circle_pos}.jpg", canvas)
                            circle_pos += 1
                            rad_list.append(r)

                    # else:
                    #     print("rmin>r>rmax")

                    # print(time.time() - start_time)

                    # if save:
                    #     print("------------ Saving required images -------------")

                        # if imsz == "full":

                    #         # cv2.imwrite(f"{save_dirx}/"+f"FULL_img_{img_name}_{exp}_0_Input.jpg",img_original)

                    #         # # cv2.imwrite(f"{save_dirx}/"+f"FULL_img_{img_name}_{exp}_1_Gblur.jpg",img_blur)
                    #         # # cv2.imwrite(f"{save_dirx}/"+f"FULL_img_{img_name}_{exp}_1_Sharp.jpg",img_sharp)

                    #         cv2.imwrite(f"{save_dirx}/"+f"FULL_img_{img_name}_{exp}_2_Threshold.jpg",imgp)
                    #         cv2.imwrite(f"{save_dirx}/"+f"FULL_img_{img_name}_{exp}_3_Canny.jpg",canny)
                    #         # # cv2.putText(canvas, f"wt: {thresh3}", (100,100), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0),2)
                    #         # # cv2.putText(canvas, f"ad_nm: {adp_th}", (100,100), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0),2)
                            # cv2.imwrite(
                            #     f"{save_dirx}/"+f"FULL_img_{img_name}_{exp}_4_Finalresult.jpg", canvas)
                # print(time.time() - start_time)

    # print(f"rad_list: {rad_list}")
    return rad_list

# ------------------------- EXECUTION AREA -------------------------


def main():

    # reading parameter json file
    data_jsonx = json.load(
        open(f"{BASE_PATH}/scripts/dimension_data.json",))
    data_jsonx = data_jsonx[0]

    # print(type(data_jsonx["c1x1"]))

    #######
    c1x1, c1x2 = data_jsonx["c1x1"], data_jsonx["c1x2"]
    c2x1, c2x2 = data_jsonx["c2x1"], data_jsonx["c2x2"]
    c3x1, c3x2 = data_jsonx["c3x1"], data_jsonx["c3x2"]
    c4x1, c4x2 = data_jsonx["c4x1"], data_jsonx["c4x2"]
    c5x1, c5x2 = data_jsonx["c5x1"], data_jsonx["c5x2"]
    c6x1, c6x2 = data_jsonx["c6x1"], data_jsonx["c6x2"]
    # c7x1, c7x2 = 1186, 1453

    c1y1, c1y2 = data_jsonx["c1y1"], data_jsonx["c1y2"]
    c2y1, c2y2 = data_jsonx["c2y1"], data_jsonx["c2y2"]
    c3y1, c3y2 = data_jsonx["c3y1"], data_jsonx["c3y2"]
    c4y1, c4y2 = data_jsonx["c4y1"], data_jsonx["c4y2"]
    c5y1, c5y2 = data_jsonx["c5y1"], data_jsonx["c5y2"]
    c6y1, c6y2 = data_jsonx["c6y1"], data_jsonx["c6y2"]
    # c7y1, c7y2 = 619, 880

    img_path = sys.argv[1]  # image input here
    # img_name = img_path.split("/")[-1].split(".")[0]

    img_o = cv2.imread(img_path, 0)
    crop_position = [(c1x1, c1y1, c1x2, c1y2), (c2x1, c2y1, c2x2, c2y2), (c3x1, c3y1, c3x2, c3y2),
                     (c4x1, c4y1, c4x2, c4y2), (c5x1, c5y1, c5x2, c5y2), (c6x1, c6y1, c6x2, c6y2)]

    circle_pos = 1
    # output_data = []
    rad_list = []
    circle_name = ["Stop Pin 1", "Stop Pin 2", "Stop Pin 3",
                   "Stop Pin 4", "Stop Pin 5", "Stop Pin 6"]
    final_result = []
    for c in crop_position:
        img = img_o[c[1]:c[3], c[0]:c[2]]
        # cv2.namedWindow("cropped", cv2.WINDOW_NORMAL)
        # while True :
        #     cv2.imshow("cropped", img)
        #     key = cv2.waitKey(1)
        #     if key == ord("q"):
        #         break

        rad = detect_circle(img, circle_pos=circle_pos,
                            rmin=data_jsonx["rmin"], rmax=data_jsonx["rmax"], adp_th=data_jsonx["adp_th"])
        circle_pos += 1
        # print(rad)
        try:
            rad_list.append(rad[0])
            # print(rad)
        except:
            pass
            # print("WORNING!!! Location mismatch!! Please place the kit perfectly.")

    # output_data.append(dict(zip(circle_name,rad_list)))
    # print(output_data)

    try:
        for i in range(len(rad_list)):
            d = {"name": circle_name[i], "dim": rad_list[i]}
            final_result.append(d)
            # print(final_result)
    except:
        pass
        # print("WORNING!!! Location mismatch!! Please place the kit perfectly.")

    print(json.dumps(final_result, indent=2))

    # working with centre circle
    # center_circle = (c7x1,c7y1,c7x2,c7y2)
    # img = img_o[center_circle[1]:center_circle[3], center_circle[0]:center_circle[2]]
    # cv2.namedWindow("cropped", cv2.WINDOW_NORMAL)
    # while True :
    #     cv2.imshow("cropped", img)
    #     key = cv2.waitKey(1)
    #     if key == ord("q"):
    #         break

    # r_s = detect_circle(img,circle_pos = circle_pos,rmin=10,rmax=1500)
    # print(f"r_s:{r_s}")
    # return
# executing main function
main()
