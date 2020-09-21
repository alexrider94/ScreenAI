import sys
import numpy as np 
from PIL import Image
import cv2 as cv
from os import listdir
from os.path import isfile, join
import tensorflow as tf

def learn(images):

    data = []
    testdata = []
    # y= kx + b
    answerPath = "./naver"
    testPath = './testdata'

    for raw in listdir(answerPath):
        image = np.array(cv.imread(join(answerPath,raw)))
        data.append(image)

    for raw in listdir(testPath):
    # The array of values
        test = np.array(cv.imread(join(testPath,raw)))
        testdata.append(test)

    files1 = [ f for f in listdir(answerPath) if isfile(join(answerPath,f))]
    # image numpy empty로 선언
    naver = np.empty(len(files1), dtype=object)
    for n in range(0,len(naver)):
        # 각 파일의 rgb값들을 images에 넣기.
        naver[n] = cv.imread(join(answerPath,files1[n]))  


    t_data = np.array(testdata)

    # y_data = np.array([[0],[0],[0],[0],[0],[1],[1],[1],[1],[1]])
    y_data = np.array([[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1]])
    x_data = np.array(data)
    # test = tf.keras.utils.to_categorical([[0,1]],num_classes=2,dtype='int')
    
    # tf.convert_to_tensor(y_data)
    print(t_data.shape)
    print(x_data.shape)
    print(y_data.shape)
    # for li in naver:
    #     print(li)

    tf.model = tf.keras.Sequential()

    tf.model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu',input_shape=[90,320,3]))
    
    tf.model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
    tf.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    tf.model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))
    tf.model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
    tf.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    tf.model.add(tf.keras.layers.Flatten())
    tf.model.add(tf.keras.layers.Dense(units=512,activation='relu'))
    tf.model.add(tf.keras.layers.Dense(units=2,activation='sigmoid'))
    #sigmod/ softmax
    tf.model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
    #categorical_crossentropy / binary_crossentropy
    tf.model.summary()

    tf.model.fit(x_data,y_data,batch_size=100,epochs=15)

    y_predict = tf.model.predict(t_data)
    print(y_predict)

def main(): 
    path = "./images"
    try:
        # f = 각각의 파일 가져오는것
        onlyfiles = [ f for f in listdir(path) if isfile(join(path,f))]
        # image numpy empty로 선언
        images = np.empty(len(onlyfiles), dtype=object)
        for n in range(0,len(images)):
            # 각 파일의 rgb값들을 images에 넣기.
            images[n] = cv.imread(join(path,onlyfiles[n]))

        #np.set_printoptions(threshold=sys.maxsize) # 값 다 보고싶을경우. 오래걸림 ^^

        #images 형태
        #1920x1080
        # [array([[[r1,g1,b1],[r2,g2,b2]....,[r1080,g1080,b1080]],[[r1081,g1081,b1081],...[r2160,g2160,b2160]],....], dtype=unit8)]
        
        # 6220800 = 1920 x 1080 x (3)rgb
        # for col in images[0]: #1080번
        #     for row in col: #1920번
        #         # print(row) [r,g,b]
        #         # print(row[0]) r
        #         # print(row[1]) g
        #         # print(row[2]) b
        #         print(row)

        learn(images)

    except IOError:
        print('there is error to open image file')
        pass

if __name__ == "__main__":
    main()