import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow import keras

# Load the MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build the neural network model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Create the GUI window
window = tk.Tk()
window.title("Number Recognition")
window.geometry("400x400")

canvas = tk.Canvas(window, width=200, height=200)
canvas.pack()

label = tk.Label(window, text="Draw a digit (0-9) in the canvas")
label.pack()

# Variable to store the drawn image
image_data = np.zeros((200, 200), dtype=np.uint8)

def predict_digit():
    # Resize the image to 28x28 and preprocess
    image = Image.fromarray(image_data)
    image = image.resize((28, 28))
    image = np.array(image) / 255.0

    # Make prediction
    prediction = model.predict(np.expand_dims(image, axis=0))
    predicted_label = np.argmax(prediction)

    # Display the predicted label
    result_label.config(text="Predicted Digit: {}".format(predicted_label))

def clear_canvas():
    # Clear the canvas and reset the image data
    canvas.delete("all")
    image_data.fill(0)
    result_label.config(text="")

def draw(event):
    x = event.x
    y = event.y
    canvas.create_oval((x - 10, y - 10, x + 10, y + 10), fill='white', outline='white')
    image_data[y - 10:y + 10, x - 10:x + 10] = 255

# Bind events to canvas
canvas.bind("<B1-Motion>", draw)

# Create buttons
predict_button = tk.Button(window, text="Predict", command=predict_digit)
predict_button.pack()

clear_button = tk.Button(window, text="Clear", command=clear_canvas)
clear_button.pack()

result_label = tk.Label(window, text="")
result_label.pack()

# Run the GUI event loop
window.mainloop()
