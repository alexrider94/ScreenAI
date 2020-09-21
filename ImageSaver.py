# print(ImageGrab.grab().size)
#(3840, 1080)

# 윈도우사용자면 화면설정에서 텍스트사이즈 및 외부 사이즈 설정을 100%로 해줘야 해상도처리 가능.

# resolution = ImageGrab.grab().size

# xSize = resolution[0]
# ySize = resolution[1]

# imageLoad = ImageGrab.grab().load()

# rgb = []
# for i in range(xSize):
#     rgb.append([])
#     for j in range(ySize):
#         rgb[i].append(ImageGrab.grab().load()[i,j])

# print(rgb)

# pix = np.array(image)

# print(pix)

import time 
import pyscreenshot as ImageGrab
import numpy as np 
import threading

def imageSave():
    # 이미지저장
    image = ImageGrab.grab()
    # sizing 처리
    # USE pillow lib image resize  16:9로
    # 내컴퓨터는 32:9  
    # 640*360, 1024*576, 1280*720, 1600*900, 1920*1080
    # resizedImage = image
    # resolution = ImageGrab.grab().size
    # print(resolution)
    
    
    #resizedImage = image.resize((1280,360))
    resizedImage = image.resize((320,90))
    timestamp = time.time()
    resizedImage.save('./images/'+str(round(timestamp)) + '.png')


def set_interval(func, sec):
    def func_wrapper():
        set_interval(func, sec)
        func()

    t = threading.Timer(sec, func_wrapper)
    t.start()
    return t
# 이미지 저장 프로세스

# 초간격
set_interval(imageSave,3)