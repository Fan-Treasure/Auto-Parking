import cv2
import os
import math
import time
from Car_Direction import cnn, method, dataset, tools
import numpy as np


def run_cmd_Popen_fileno(cmd_string):
    """
    执行cmd命令，并得到执行后的返回值，python调试界面输出返回值
    :param cmd_string: cmd命令，如：'adb devices'
    :return:
    """
    import subprocess

    print('运行cmd指令：{}'.format(cmd_string))
    return subprocess.Popen(cmd_string, shell=True, stdout=None, stderr=None).wait()

# 运行此程序需要接摄像头及小车
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # 打开摄像头
sig = 0
while(1):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示
    cv2.imwrite("pic/test.png", frame)  # 保存路径
    run_cmd_Popen_fileno('CUDA_VISIBLE_DEVICES=0')
    run_cmd_Popen_fileno('python -u PaddleDetection/tools/infer.py -c PaddleDetection/configs/ppyolo/ppyolov2_r50vd_dcn_voc.yml -o weights=Models/model_final --save_txt output --infer_dir pic')
    while(os.path.exists('output/test.txt')):
        with open('output/test.txt','r') as f:
            data =f.readline()
        with open('Labels/test.txt','w') as x:
            message = 'pic/test.png'+' ' +data.split()[2]+' '+ data.split()[3]+' '+ data.split()[4]+' '+ data.split()[5] + ' 0'
            x.write(message)
        TEST_PATH = "Labels/test.txt"
        test_dataset = dataset.Dataset(TEST_PATH, batch_size=1, mode='test')
        prs = method.test('Models/orientation_detection_model.pdparams', test_dataset)
        print(prs)
        pic = cv2.imread("pic/test.png")
        w = pic.shape[1]
        h = pic.shape[0]
        cv2.circle(pic,[int(float(data.split()[2])+float(data.split()[4])/2),int(float(data.split()[3])+float(data.split()[5])/2)],50,[255,255,255],3)
        cv2.line(pic,[int(float(data.split()[2])+float(data.split()[4])/2),int(float(data.split()[3])+float(data.split()[5])/2)],[int(float(data.split()[2])+float(data.split()[4])/2-50*math.cos(prs[0]/180*math.pi)),int(float(data.split()[3])+float(data.split()[5])/2-50*math.sin(prs[0]/180*math.pi))],[255,255,255],3)
        cv2.imshow('0',pic)
        cv2.waitKey(0)
        os.remove('pic/test.png')
        os.remove('output/test.png')
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按‘q’退出
        break
cap.release()
cv2.destroyAllWindows()