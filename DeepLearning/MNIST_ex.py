import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 as cv

(train_images, train_labels),(test_images, test_labels) = tf.keras.datasets.mnist.load_data()

print(train_images.shape)
print(test_images.shape)
# plt.imshow(train_images[0], cmap='Greys')
# plt.show()

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=512, activation='relu', input_shape=(28*28,)))
model.add(tf.keras.layers.Dense(units=10,activation='sigmoid'))

model.compile(optimizer='RMSprop', loss='mse', metrics=['accuracy'])

########### 전처리 (flattening) ###########
train_images = train_images.reshape(60000,784)
train_images = train_images.astype('float32') / 255.0

test_images = test_images.reshape(10000,784)
test_images = test_images.astype('float32') / 255.0

######## ONE-HOT-ENCODING #########
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

history = model.fit(train_images, train_labels, epochs=10, batch_size=128)
loss = history.history['loss']
acc = history.history['accuracy']
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, acc, 'r', label='Accuracy')
plt.xlabel('epochs')
plt.ylabel('loss/acc')
plt.show()

image = cv.imread('test1.png', cv.IMREAD_GRAYSCALE)
image = cv.resize(image, (28, 28))
image = image.astype('float32')
image = image.reshape(1, 784)
image = 255-image # 이미지 색상 반전
image /= 255.0 # 이미지 값을 0~1 사이의 값으로 변경
plt.imshow(image.reshape(28, 28),cmap='Greys')
plt.show()

pred = model.predict(image.reshape(1, 784), batch_size=1)
print("추정된 숫자=", pred.argmax())
