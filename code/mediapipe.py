import cv2
import mediapipe as mp
import time
import numpy as np


# 计算角度
def calculate_angle(a, b, c):
    """
    :param a: 关键点a
    :param b: 关键点b
    :param c: 关键点c
    :return:
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    # 弧度转角度
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle

    return angle


# 计算欧式距离
def calculate_dist(a, b):
    '''
    计算欧式距离
    :param a:
    :param b:
    :return:
    '''
    a = np.array(a)
    b = np.array(b)
    # 2范数
    dist = np.linalg.norm(a - b)
    return dist


cap = cv2.VideoCapture('08.MP4')
# cap = cv2.VideoCapture('E:\\cv\\28.mp4')
# cap = cv2.VideoCapture('rtsp://admin:admin@10.42.0.218:8554/live')
# cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,  # 是静态图片还是连续视频帧
                      max_num_hands=1,  # 最多检测几只手
                      min_detection_confidence=0.6)  # 置信度阈值
mpDraw = mp.solutions.drawing_utils  # 绘制手部点函数
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=1)  # 改变点的颜色,粗度
handConStyle = mpDraw.DrawingSpec(color=(255, 255, 255), thickness=2)  # 改变线的颜色，粗度
pTime = 0
cTime = 0

Pic_num = 0

FLATVIDEO = True


while True:
    ret, img = cap.read()
    try:
        imgHeight, imgWidth = img.shape[:2]  # 获取图像大小
        
        img_rule_width = 640  # 限定640
        img = cv2.resize(img, (img_rule_width, int(imgHeight / (imgWidth / img_rule_width))))  # 限定宽度

        if FLATVIDEO:
            FLATVIDEO = False
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 视频编解码器
            fps = cap.get(cv2.CAP_PROP_FPS)
            out = cv2.VideoWriter(f'08.avi', fourcc, fps, (imgWidth, imgHeight))  # 写入视频
    except:
        break
    if ret:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将初始bgr图片转为rgb图片
        imgHeight, imgWidth = img.shape[:2]  # 获取图像大小
        result = hands.process(imgRGB)  # 识别当前的图像

        zeros1 = np.zeros((imgHeight,imgWidth), dtype=np.uint8)                                  # 创建空白背景
        zeros1 = cv2.cvtColor(zeros1,cv2.COLOR_GRAY2BGR)                                         # 转BGR图像
        zeros2 = zeros1.copy()

        if result.multi_hand_landmarks:  # 检测到手部
            for handLms in result.multi_hand_landmarks:  # 遍历每一只手
                Hand_x = [data.x for data in handLms.landmark]  # 所有关键点的X坐标
                Hand_y = [data.y for data in handLms.landmark]  # 所有关键点的Y坐标
                Hand_z = [data.z for data in handLms.landmark]  # 所有关键点的Z坐标
                # mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS,handLmsStyle,handConStyle)# 画出骨骼

                # 提取关键点
                point5 = [int(imgWidth * Hand_x[5]), int(imgHeight * Hand_y[5])]
                point6 = [int(imgWidth * Hand_x[6]), int(imgHeight * Hand_y[6])]
                # point7 = [int(imgWidth * Hand_x[7]), int(imgHeight * Hand_y[7])]
                point8 = [int(imgWidth * Hand_x[8]), int(imgHeight * Hand_y[8])]
                point4 = [int(imgWidth * Hand_x[4]), int(imgHeight * Hand_y[4])]
                point3 = [int(imgWidth * Hand_x[3]), int(imgHeight * Hand_y[3])]
                point2 = [int(imgWidth * Hand_x[2]), int(imgHeight * Hand_y[2])]
                point1 = [int(imgWidth * Hand_x[1]), int(imgHeight * Hand_y[1])]
                point0 = [int(imgWidth * Hand_x[0]), int(imgHeight * Hand_y[0])]

                # 绘制5/6/8关键点
                # v2.circle(img,point5,4,(255,255,0),cv2.FILLED)
                cv2.circle(img, point5, 4, (255, 255, 0), cv2.FILLED)
                cv2.circle(img, point6, 4, (255, 255, 0), cv2.FILLED)
                cv2.circle(img, point8, 4, (255, 255, 0), cv2.FILLED)
                cv2.circle(img, point4, 4, (255, 255, 0), cv2.FILLED)
                cv2.circle(img, point0, 4, (255, 255, 0), cv2.FILLED)

                cv2.circle(zeros1, point5, 4, (255, 255, 0), cv2.FILLED)
                cv2.circle(zeros1, point6, 4, (255, 255, 0), cv2.FILLED)
                cv2.circle(zeros1, point8, 4, (255, 255, 0), cv2.FILLED)
                cv2.circle(zeros1, point4, 4, (255, 255, 0), cv2.FILLED)
                cv2.circle(zeros1, point0, 4, (255, 255, 0), cv2.FILLED)

                cv2.circle(zeros2, point5, 4, (255, 255, 0), cv2.FILLED)
                cv2.circle(zeros2, point6, 4, (255, 255, 0), cv2.FILLED)
                cv2.circle(zeros2, point8, 4, (255, 255, 0), cv2.FILLED)
                cv2.circle(zeros2, point4, 4, (255, 255, 0), cv2.FILLED)
                cv2.circle(zeros2, point0, 4, (255, 255, 0), cv2.FILLED)

                # 连接关键点
                cv2.line(img, point5, point6, (0, 255, 0), 2)
                # cv2.line(img, point7, point6, (0, 255, 0), 2)
                cv2.line(img, point6, point8, (0, 255, 0), 2)
                cv2.line(img, point4, point3, (0, 255, 0), 2)
                cv2.line(img, point3, point1, (0, 255, 0), 2)
                cv2.line(img, point1, point0, (0, 255, 0), 2)
                cv2.line(img, point5, point0, (0, 255, 0), 2)

                cv2.line(zeros1, point5, point6, (0, 255, 0), 2)
                # cv2.line(zeros1, point7, point6, (0, 255, 0), 2)
                cv2.line(zeros1, point6, point8, (0, 255, 0), 2)
                cv2.line(zeros1, point4, point3, (0, 255, 0), 2)
                cv2.line(zeros1, point3, point1, (0, 255, 0), 2)
                cv2.line(zeros1, point1, point0, (0, 255, 0), 2)
                cv2.line(zeros1, point5, point0, (0, 255, 0), 2)

                cv2.line(zeros2, point5, point6, (0, 255, 0), 2)
                # cv2.line(zeros2, point7, point6, (0, 255, 0), 2)
                cv2.line(zeros2, point6, point8, (0, 255, 0), 2)
                cv2.line(zeros2, point4, point3, (0, 255, 0), 2)
                cv2.line(zeros2, point3, point1, (0, 255, 0), 2)
                cv2.line(zeros2, point1, point0, (0, 255, 0), 2)
                cv2.line(zeros2, point5, point0, (0, 255, 0), 2)


                # 得到角度
                # Key_angle = calculate_angle(point5, point6, point8)
                # cv2.putText(img, f"Angle:{int(Key_angle)}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
                # cv2.putText(zeros1,f"Angle:{int(Key_angle)}",(30,90),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),3)

                # 得到距离
                #Key_distance = calculate_dist(point6, point8)
                #cv2.putText(img, f"Distance:{int(Key_distance)}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255),3)
                #cv2.putText(zeros1,f"Distance:{int(Key_distance)}",(30,130),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),3)

                cv2.imwrite(f'70/{Pic_num}.jpg',zeros2) # 保存Pic_num
                Pic_num += 1

        # 将一秒多少帧显示在图片上
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
       # cv2.putText(img, f"FPS:{int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
        #cv2.putText(zeros1,f"FPS:{int(fps)}",(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),3)
        res = np.hstack((img,zeros1))
        cv2.imshow('识别结果', res)

        out.write(cv2.resize(zeros1, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))))# 写入帧
        #out.write(cv2.resize(res,(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))))  # 写入帧
    # print(int(Key_angle))
    # print(int(Key_distance))

    

    if cv2.waitKey(1) == ord('q'):
        break

out.release()
