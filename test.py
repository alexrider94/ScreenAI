# import numpy as np
# import tensorflow as tf
# import random

# mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train)
# print(y_train)


# x_test = x_test / 255
# x_train = x_train / 255
# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# print(x_train)

# # one hot encode y data
# y_train = tf.keras.utils.to_categorical(y_train, 10)
# y_test = tf.keras.utils.to_categorical(y_test, 10)

# # hyper parameters
# learning_rate = 0.001
# training_epochs = 12
# batch_size = 128

# tf.model = tf.keras.Sequential()
# # L1
# tf.model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
# tf.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# # L2
# tf.model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
# tf.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# # L3 fully connected
# tf.model.add(tf.keras.layers.Flatten())
# tf.model.add(tf.keras.layers.Dense(units=10, kernel_initializer='glorot_normal', activation='softmax'))

# tf.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
# tf.model.summary()

# tf.model.fit(x_train, y_train, batch_size=batch_size, epochs=training_epochs)

# # predict 10 random hand-writing data
# y_predicted = tf.model.predict(x_test)
# for x in range(0, 10):
#     random_index = random.randint(0, x_test.shape[0]-1)
#     print("index: ", random_index,
#           "actual y: ", np.argmax(y_test[random_index]),
#           "predicted y: ", np.argmax(y_predicted[random_index]))

# evaluation = tf.model.evaluate(x_test, y_test)
# print('loss: ', evaluation[0])
# print('accuracy', evaluation[1])

import tensorflow as tf

x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=2))
# use sigmoid activation for 0~1 problem
tf.model.add(tf.keras.layers.Activation('sigmoid'))

''' 
better result with loss function == 'binary_crossentropy', try 'mse' for yourself
adding accuracy metric to get accuracy report during training
'''
tf.model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.01), metrics=['accuracy'])
tf.model.summary()

history = tf.model.fit(x_data, y_data, epochs=5000)

# Accuracy report
print("Accuracy: ", history.history['accuracy'][-1])