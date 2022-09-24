# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import random
import time

import cv2 as cv
import numpy as np

# cap = cv.VideoCapture("./video2.mp4")
cap = cv.VideoCapture(0)

colorLow = np.array([156, 100, 46])
colorHigh = np.array([180, 255, 255])

# 记录目标位置的位置。
attacks_count = 15
attacks_speed = 3
attacks_location = []
for i in range(attacks_count):
    # 前3个为当前x y z，后3个为目标x y z
    attacks_location.append([-1, -1, -1, -1])


def attack_move(location, speed, video_width, video_height):
    if location[0] == location[2] and location[1] == location[3]:
        return [random.randint(0, video_width), 0, random.randint(0, video_width),
                0]
    if location[0] > location[2]:
        if location[0] - speed < location[2]:
            location[0] = location[2]
        else:
            location[0] = location[0] - speed
    else:
        if location[0] + speed > location[2]:
            location[0] = location[2]
        else:
            location[0] = location[0] + speed

    if location[1] > location[3]:
        if location[1] - speed < location[3]:
            location[1] = location[3]
        else:
            location[1] = location[1] - speed
    else:
        if location[1] + speed < location[3]:
            location[1] = location[3]
        else:
            location[1] = location[1] + speed
    return location


# 选择一个动物去追逐
logo_img = cv.imread("./logo.jpg")

# 用于开始初始化参数
game_start_init = False
game_over = False
# 初始化参数
x, y, w, h = -1, -1, -1, -1
width, height = -1, -1
while cap.isOpened():
    if not game_start_init:
        # 获取输入的宽度和高度
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        for i in range(attacks_count):
            # 前3个为当前x y z，后3个为目标x y z
            attacks_location.append([random.randint(0, width), 0, random.randint(0, width),
                                     random.randint(0, height)])
        game_start_init = True
    ret, image = cap.read()
    if ret:
        image = cv.flip(image, 1)
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        mask_img = cv.inRange(hsv_image, colorLow, colorHigh)
        contours, hierarchy = cv.findContours(mask_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        max_area_contours_index = -1
        max_area = -1
        for i in range(len(contours)):
            area = cv.contourArea(contours[i])
            if area < 500:
                continue
            x, y, w, h = cv.boundingRect(contours[i])
            # 查找最大的矩形红色面积
            if area > 500 and w > 20 and h > 20 and 1.6 > w / h > 0.3:
                if area > max_area:
                    max_area = area
                    max_area_contours_index = i
                # cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if max_area_contours_index != -1:
            x, y, w, h = cv.boundingRect(contours[max_area_contours_index])
            # 更新攻击求位置
            for i in range(attacks_count):
                # 前3个为当前x y z，后3个为目标x y z
                attacks_location[i] = attack_move(attacks_location[i], speed=attacks_speed, video_width=width,
                                                  video_height=height)
                cv.circle(image, (attacks_location[i][0], attacks_location[i][1]), 10, (255, 255, 0))
                if attacks_location[i][0] >= x and attacks_location[i][0] <= x + w and attacks_location[i][1] >= y and \
                        attacks_location[i][1] <= y + h:
                    # 游戏结束
                    cv.putText(image, 'Game Over', (100, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
                    cv.imshow("aaa", image)
                    key = cv.waitKey(5000)
                    game_over = True
                    break
            if game_over:
                break
            # 画出Logo位置
            image[y:y + h, x:x + w, :] = cv.resize(logo_img, (w, h), interpolation=cv.INTER_NEAREST)
        else:
            if x != -1:
                image[y:y + h, x:x + w, :] = cv.resize(logo_img, (w, h), interpolation=cv.INTER_NEAREST)
            cv.putText(image, 'Game Pause', (100, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

        cv.imshow("aaa", image)

    key = cv.waitKey(1)
    if key == 27:
        break
cap.release()
cv.destroyWindow("aaa")
