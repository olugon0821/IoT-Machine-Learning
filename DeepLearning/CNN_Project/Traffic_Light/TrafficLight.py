import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras.optimizers


# RED Light 리스트 생성
Redlight_images = list()
for i in range(140):
    file = "./red/" + "{0:02d}.JPG".format(i+1)
    img = cv2.imread(file)
    if img is None:
        print("파일이 없습니다")
        break
    # 이미지 사이즈 조정
    img = cv2.resize(img,(256,256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    Redlight_images.append(img)

    # 이미지 리스트 출력
    def plot_images(n_row:int, n_col:int, images:list[np.ndarray]) -> None:
        fig = plt.figure()
        (fig,ax) = plt.subplots(n_row, n_col, figsize=(n_col, n_row))
        for i in range(n_row):
            for j in range(n_col):
                axis = ax[i,j]
                axis.get_xaxis().set_visible(False)
                axis.get_yaxis().set_visible(False)
                axis.imshow(images[i * n_col + j])
        plt.show()
        return None

# plot_images(n_row=3, n_col=9, images=Redlight_images)

# GREEN Light 리스트 생성
Greenlight_images = list()
for i in range(140):
    file = "./green/" + "{0:02d}.JPG".format(i+1)
    img = cv2.imread(file)
    if img is None:
        print("파일이 없습니다")
        break
    img = cv2.resize(img,(256,256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    Greenlight_images.append(img)

# plot_images(n_row=1, n_col=17, images=Greenlight_images)

# YELLOW Light 리스트 생성
Yellowlight_images = list()
for i in range(80):
    file = "./yellow/" + "{0:02d}.JPG".format(i+1)
    img = cv2.imread(file)
    if img is None:
        print("파일이 없습니다")
        break
    img = cv2.resize(img,(256,256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    Yellowlight_images.append(img)

# plot_images(n_row=2, n_col=8, images=Yellowlight_images)

# X_train data 생성
X = Redlight_images + Greenlight_images + Yellowlight_images
y = [[1,0,0]]*len(Redlight_images) + [[0,1,0]]*len(Greenlight_images) + [[0,0,1]]*len(Yellowlight_images)

# CNN NETWORK 생성
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(input_shape=(64,64,3), kernel_size=(3,3), filters=32),
#     tf.keras.layers.MaxPooling2D((2,2), strides=2),
#     tf.keras.layers.Conv2D(kernel_size=(3,3), filters=32),
#     tf.keras.layers.MaxPooling2D((2,2), strides=2),
#     tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=32),
#     tf.keras.layers.MaxPooling2D((2, 2), strides=2),
#     tf.keras.layers.Flatten(),
#
#     ## Neural Network ##
#     tf.keras.layers.Dense(units=512, activation='relu'),
#     tf.keras.layers.Dense(units=128, activation='relu'),
#     tf.keras.layers.Dense(units=16, activation='relu'),
#     tf.keras.layers.Dense(units=3, activation='softmax')
# ])

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(input_shape=(256,256, 3), kernel_size=(3, 3), filters=64),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=64),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=64),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Flatten(),

    ## Neural Network ##
    tf.keras.layers.Dense(units=512, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
    tf.keras.layers.Dense(units=128, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
    tf.keras.layers.Dense(units=64, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
    tf.keras.layers.Dense(units=32, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
    tf.keras.layers.Dense(units=16, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
    tf.keras.layers.Dense(units=3, activation='softmax')
])

model.summary()

model.compile(optimizer='Adam',loss='categorical_crossentropy', metrics=['accuracy'])

X = np.array(X)
y = np.array(y)

history = model.fit(x = X, y = y, epochs=3500)

# 테스트 리스트
examples_images = list()
for i in range(10):
    file = "./ex/" + "{0:02d}.JPG".format(i+1)
    img = cv2.imread(file)
    if img is None:
        print("파일이 없습니다")
        break
    img = cv2.resize(img,(64,64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    examples_images.append(img)

examples_images = np.array(examples_images)
plot_images(2,5,examples_images)
predict_images = model.predict(examples_images)
print(predict_images)

model.save('TL7.h5')