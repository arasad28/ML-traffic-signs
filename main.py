import matplotlib.pyplot as plt
import os
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
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH = "/"
train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Training")
test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Testing")

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

plt.show()
images28 = [transform.resize(image, (28, 28)) for image in images]



images28 = np.array(images28)

images28 = rgb2gray(images28)
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images28[traffic_signs[i]], cmap="gray")
    plt.subplots_adjust(wspace=0.5)
    

plt.show()

x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28])
y = tf.placeholder(dtype = tf.int32, shape = [None])


images_flat = tf.contrib.layers.flatten(x)

logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)


loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, 

train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)


correct_pred = tf.argmax(logits, 1)


accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.set_random_seed(1234)
sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(201):
        print('EPOCH', i)
        _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images28, y: labels})
        if i % 10 == 0:
            print("Loss: ", loss)
        print('DONE WITH EPOCH')

sample_indexes = random.sample(range(len(images28)), 10)
sample_images = [images28[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]


predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]
                        


fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
             fontsize=12, color=color)
    plt.imshow(sample_images[i],  cmap="gray")

plt.show()