import numpy as np
import tensorflow as tf
# Model 1 trained on Squared
x_train = np.random.rand(10000);
y_train = x_train**2;
x_test = np.random.rand(1000);
y_test = x_test**2;

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(5, activation=tf.nn.relu, input_dim = 1),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
model.compile(optimizer='sgd',
              loss='mean_squared_error',
              metrics=['accuracy'])

print("Prediction 1 of .5 = ", model.predict(np.array([0.5])))
model.fit(x_train, y_train, epochs=50)
model.evaluate(x_test, y_test)
print("Prediction 1 of .5 = ", model.predict(np.array([0.5])))

# Model 2 trained on Model 1
x_train = np.random.rand(10000);
y_train = model.predict(x_train);
x_test = np.random.rand(1000);
y_test = model.predict(x_test);

model2 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(5, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
model2.compile(optimizer='sgd',
               loss='mean_squared_error',
               metrics=['accuracy'])

model2.fit(x_train, y_train, epochs=50)
model2.evaluate(x_test, y_test)
print("Prediction 2 of .5 = ", model2.predict(np.array([0.5])))
