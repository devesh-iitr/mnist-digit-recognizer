import tensorflow as tf
print(tf.__version__)

(x_train, y_train), (x_test, y_test)= tf.keras.datasets.mnist.load_data()
print("Training Images:", x_train.shape)
print("Training labels:", y_train.shape)
print("Test Image:", x_test.shape)

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
plt.imshow(x_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.axis('off')
plt.show()

fig, axes = plt.subplots(2,5, figsize=(10,5))

for i, ax in enumerate(axes.flat):
  ax.imshow(x_train[i], cmap='gray')
  ax.set_title(f"{y_train[i]}")
  ax.axis('off')

plt.show()

print("One image as numbers:")
print(x_train[0])
print("\nMin pixel value:", x_train[0].min())
print("max pixel value:", x_train[0].max())


x_train = x_train/255.0
x_test = x_test/255.0

print("Min value now:", x_train[0].min())
print("Max value now:", x_train[0].max())


model_cnn = tf.keras.Sequential([
    tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model_cnn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model_cnn.summary()

history_cnn = model_cnn.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)

test_loss, test_acc = model_cnn.evaluate(x_test, y_test, verbose=0)
print(f"CNN Test accuracy: {test_acc:.4f}")

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history_cnn.history['accuracy'], label='Train')
plt.plot(history_cnn.history['val_accuracy'], label='Validation')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history_cnn.history['loss'], label='Train')
plt.plot(history_cnn.history['val_loss'], label='Validation')
plt.title('loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend()

plt.tight_layout()
plt.show()

import numpy as np

index=69
img= x_test[index]
true_label= y_test[index]

prediction=model_cnn.predict(img[np.newaxis, ...], verbose=0)
predicted_digit= np.argmax(prediction)
confidence= prediction[0][predicted_digit]*100

plt.imshow(img, cmap='gray')
plt.title(f"Predicted: {predicted_digit} | True: {true_label} | Confidence: {confidence:.1f}%")
plt.axis('off')
plt.show()

print("Probabilities for all digits:")
for i,prob in enumerate(prediction[0]):
  bar= '█' * int(prob * 40)
  print(f" {i}: {bar} {prob*100:.2f}%")


from google.colab import files
uploaded = files.upload()

import numpy as np
from PIL import Image

filename= list(uploaded.keys())[0]

img= Image.open(filename).convert('L')
img= img.resize((28,28))

img_array= np.array(img)

img_array= 255-img_array

img_array = img_array/ 255.0

img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())


plt.imshow(img_array, cmap='gray')
plt.title("What the model sees")
plt.axis('off')
plt.show()


prediction = loaded_model.predict(img_array[np.newaxis, ...], verbose=0)
predicted_digit = np.argmax(prediction)
confidence = prediction[0][predicted_digit] * 100

print(f"\nPredicted digit: {predicted_digit}")
print(f"Confidence: {confidence:.1f}%")

print("\nProbabilities for each digit:")
for i, prob in enumerate(prediction[0]):
    bar = '█' * int(prob * 40)
    print(f"  {i}: {bar} {prob*100:.2f}%")
