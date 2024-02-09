import matplotlib.pyplot as plt
import os
import random
from skimage import data, transform,rgb2gray
import tensorflow as tf 
import random
import numpy as np

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.io.imread(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH = "/img"
train_data_directory = os.path.join(ROOT_PATH, "Training")
test_data_directory = os.path.join(ROOT_PATH, "Testing")

images, labels = load_data(train_data_directory)


unique_labels = set(labels)
plt.figure(figsize=(15, 15))

i = 1


for label in unique_labels:
    image = images[labels.index(label)]
    plt.subplot(8, 8, i)
    plt.axis('off')
    plt.title("Label {0} ({1})".format(label, labels.count(label)))
    i += 1
    plt.imshow(image)

# plt.show()
images28 = [transform.resize(image, (28, 28)) for image in images]


images28 = np.array(images28)
images28 = rgb2gray(images28)


tf.random.set_seed(1234)
np.random.seed(1234)
images28 = np.random.randn(100, 28, 28)
labels = np.random.randint(0, 62, size=(100,))
x = tf.keras.Input(shape=(28, 28), dtype=tf.float32, name="input_images")
y = tf.keras.Input(shape=(), dtype=tf.int32, name="input_labels")
images_flat = tf.keras.layers.Flatten()(x)
logits = tf.keras.layers.Dense(62, activation=tf.nn.relu)(images_flat)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
optimizer = tf.optimizers.Adam(learning_rate=0.001)
model = tf.keras.Model(inputs=x, outputs=logits)
for i in range(201):
    print('EPOCH', i)

    with tf.GradientTape() as tape:
        logits = model(images28, training=True)  # Ensure the model is in training mode
        current_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    grads = tape.gradient(current_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    correct_pred = tf.argmax(logits, axis=1)
    accuracy_val = tf.reduce_mean(tf.cast(tf.equal(correct_pred, tf.cast(labels, tf.int64)), tf.float32))

    if i % 10 == 0:
        print("Loss: ", tf.keras.backend.eval(current_loss))
        print("Accuracy: {:.2%}".format(tf.keras.backend.eval(accuracy_val)))

    print('DONE WITH EPOCH')





images28 = np.random.randn(100, 28, 28)
labels = np.random.randint(0, 62, size=(100,))

sample_indexes = random.sample(range(len(images28)), 10)
sample_images = [images28[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

predicted = model.predict(np.array(sample_images))
predicted_labels = np.argmax(predicted, axis=1)

print("Real Labels:", sample_labels)
print("Predicted Labels:", predicted_labels)

fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted_labels[i]
    plt.subplot(5, 2, 1 + i)
    plt.axis('off')
    color = 'green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
             fontsize=12, color=color)
    plt.imshow(sample_images[i], cmap="gray")
plt.show()