import cv2
import tensorflow as tf
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import keras.optimizers

# 사람의 얼굴을 list
face_images = list()
for i in range(15):
    file = "./faces/" + "img{0:02d}.jpg".format(i+1)
    img = cv2.imread(file)
    if img is None:
        print("파일이 없습니다")
        break
    img = cv2.resize(img,(64,64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_images.append(img)

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

plot_images(n_row=3, n_col=5, images=face_images)

animals_images = list()
for i in range(15):
    file = "./animals/" + "img{0:02d}.jpg".format(i+1)
    img = cv2.imread(file)
    if img is None:
        print("파일이 없습니다")
        break
    img = cv2.resize(img,(64,64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    animals_images.append(img)

plot_images(n_row=3, n_col=5, images=animals_images)

# X_train data 생성
X = face_images + animals_images
y = [[1,0]]*len(face_images) + [[0,1]]*len(animals_images)

# CNN NETWORK 생성
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(input_shape=(64,64,3), kernel_size=(3,3), filters=32),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),
    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=32),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),
    tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=32),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Flatten(),

    ## Neural Network ##
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=2, activation='softmax')
])

model.summary()

model.compile(optimizer='Adam',loss='categorical_crossentropy', metrics=['accuracy'])

X = np.array(X)
y = np.array(y)

history = model.fit(x = X, y = y, epochs=1000)

# 테스트 리스트
examples_images = list()
for i in range(10):
    file = "./examples/" + "img{0:02d}.jpg".format(i+1)
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

fig = plt.Figure()
(fig,ax) = plt.subplots(2,5,figsize=(10,4))
for i in range(2):
    for j in range(5):
        axis = ax[i,j]
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
        if predict_images[i * 5 + j][0] > 0.6:
            axis.imshow(examples_images[i * 5 + j])
plt.show()
