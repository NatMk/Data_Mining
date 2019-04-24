# import tensorflow library
import tensorflow as tf

# tensorflow library has many datasets, we will work with mnist
# MNIST handwritten digits dataset
mnist = tf.keras.datasets.mnist

# using load_data() function, we can load data for further work
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# All images has 256 colors (from 0 to 255, where 0 - black, 255 - white)
# But for any model, working with large numbers is more difficult
# because it's hard to find the necessary regularizers
# So we normalize data
# Now, all numbers are from 0 to 1
x_train, x_test = x_train/255.0, x_test/255.0

# Creating neural network
# We create our own NN, for this, firstly we must create object which can keep all layers (Sequential).
# Then add Flatten() layers - flatten our feature map into a column. The flattening step is that we have long vector of input data that we processed further.
# The next is Dense layers. Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer 
# Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.
# And finale Dense with 10 result

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
	#hidden nodes updated to 10 units (neurons)
    tf.keras.layers.Dense(10, activation = tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation = tf.nn.softmax)
])

# Compile our model with Adam optimizer and sparse_categorical_crossentropy losses
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# training model
model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)

from matplotlib import pyplot as plot

hidden_layers = model.layers[1].get_weights()[0]

#Transpose the matrix to get the plof of 10X784
hidden_layers_Transposed = hidden_layers.T

#Loop through each weight in the first hidden layer) to the image dimension (in #2D) and display images
index = 0
for weight in hidden_layers_Transposed:
	plot.figure()
	image = hidden_layers_Transposed [index].reshape(28, 28)
	plot.imshow(image)
	plot.show()
	index = index + 1


