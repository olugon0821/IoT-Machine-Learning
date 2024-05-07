import keras.optimizers
import tensorflow as tf

model = tf.keras.models.Sequential() # API

layer1 = tf.keras.layers.Dense(units=2, input_shape=(2,), activation='sigmoid')
model.add(layer1)
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(learning_rate=0.2))

X = tf.Variable([[0,0],[0,1],[1,0],[1,1]])
y = tf.Variable([0, 1, 1, 0])

model.fit(X, y, batch_size=1,epochs=10000)

print(model.predict(X))
