import random
import numpy as np
import cv2
import numpy
import tensorflow as tf
import os
from mss import mss
import winsound
from model import model

train = False
checkpoint_id = 60000
test = False


bounds = {'top': 600, 'left': 1200, 'width': 1400, 'height': 1000}

newest_checkpoint = f"Checkpoints/First/{checkpoint_id}"

batch_size = 64
epochs = 30
batch_multiplyer = 1
batch_shape = (batch_size, bounds["height"], bounds["width"], 1)
amount_toot = len(next(os.walk("../images/toot/"))[2])
amount_notoot = len(next(os.walk("../images/notoot/"))[2])
i = 0
batches_per_epoch = amount_notoot // batch_size
total_batches = batches_per_epoch * epochs



loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)


def normalize_image(im):
    im = numpy.array(im, dtype=numpy.uint8)
    im = im[bounds["top"]:bounds["top"] + bounds["height"], bounds["left"]:bounds["left"] + bounds["width"]]
    im = numpy.flip(im[:, :, :3], 2)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = im / 255
    return im


def generate_batch(toot=False):
    train_ds = []
    label = 0
    path = "../images/notoot/"
    if toot:
        path = "../images/toot/"
        label = 1
    while True:
        for dirpath, dirnames, filenames in os.walk(path):
            random.shuffle(filenames)
            for filename in filenames:
                screen_img = cv2.imread(f"{path}{filename}")
                im = normalize_image(screen_img)
                train_ds.append(im)
                if len(train_ds) == batch_size:
                    train_tensor = tf.convert_to_tensor(train_ds, dtype=tf.float32)
                    train_tensor = tf.reshape(train_tensor, shape=batch_shape)
                    yield train_tensor, tf.convert_to_tensor([label] * batch_size,
                                                             dtype=tf.uint8)
                    train_ds = []


def generate_combined_dataset(gen_a, gen_b, size=batch_size * batch_multiplyer):
    data = []
    label = []
    while len(data) < size:
        dataA, labelA = next(gen_a)
        dataB, labelB = next(gen_b)
        data.extend(dataA)
        label.extend(labelA)
        data.extend(dataB)
        label.extend(labelB)
    x = list(zip(data, label))
    random.shuffle(x)
    data, label = zip(*x)
    return tf.convert_to_tensor(data), tf.convert_to_tensor(label)


def loss(x, y, training):
    y_ = model(x, training=training)
    return loss_object(y_true=y, y_pred=y_)


def grad(inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def do_batch():
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    data, label = generate_combined_dataset(gen_toot, gen_notoot)
    x = data[:len(data) * 10 // 7]
    y = label[:len(data) * 10 // 7]
    val_x = data[len(data) * 10 // 7:]
    val_y = label[len(data) * 10 // 7:]
    loss_value, grads = grad(data, label)
    #print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(), loss_value.numpy()))
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    #print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(), loss(data, label, training=True).numpy()))
    epoch_loss_avg.update_state(loss_value)
    epoch_accuracy.update_state(y, model(x, training=True))
    return epoch_loss_avg.result(), epoch_accuracy.result()
    # history = model.fit(x=data, y=label, batch_size=batch_size, epochs=1)


gen_notoot = generate_batch()
gen_toot = generate_batch(toot=True)
train_loss_results = []
train_accuracy_results = []
times_to_train = total_batches // batch_multiplyer
for i in range(times_to_train):
    epoch_loss_avg, epoch_accuracy = do_batch()
    train_loss_results.append(epoch_loss_avg)
    train_accuracy_results.append(epoch_accuracy)
    if i % 10 == 0:
        print(f"-----------Batch {i} von {total_batches // batch_multiplyer}-----------")
        print("Loss: {:.3f}, Accuracy: {:.3%}".format(epoch_loss_avg, epoch_accuracy))
    if not train:
        model.load_weights(newest_checkpoint)
        break
    if i == 0:
        #model.load_weights(newest_checkpoint)
        continue
    if i % 10 == 0:
        model.save_weights(f".\\Checkpoints\\First\\{checkpoint_id + 100 + i}")
print(model.summary())

if train:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(train_loss_results)

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(train_accuracy_results)
    plt.show()

def get_evaluation_image(show=True):
    train_ds = []
    label = 0
    screen = mss()
    while True:
        screen_img = screen.grab({"top": 0, "left": 0, "width": 3840, "height": 2160})
        if show:
            cv2.imshow('screen', np.array(screen_img))
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                cv2.destroyAllWindows()
        im = normalize_image(screen_img)
        im_tensor = tf.convert_to_tensor(im, dtype=tf.float32)
        im_tensor = tf.reshape(im_tensor, shape=(1, 1000, 1400, 1))
        yield im_tensor

def result(res_vec):
    print(res_vec.numpy()[0][1])
    if res_vec.numpy()[0][1] > 0.95:
        return True # TOOOT
    else:
        return False # NO TOOT

def you_fucked_up(bearer_token = None):
    if test:
        file = "../ES_Sci Fi Beep Error - SFX Producer.wav"
        winsound.PlaySound(file, winsound.SND_FILENAME)
        return
    ctrlx_connection.motion_trigger(bearer_token)

if not train:
    import ctrlx_connection
    bearer_token = ctrlx_connection.get_token()

    image_generator = get_evaluation_image(show=False)
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    while False:
        data, label = generate_combined_dataset(gen_toot, gen_notoot)
        model.evaluate(data, label)

    sequence = [False] * 3
    while True:
        res = result(model(next(image_generator)))
        if all(sequence) and res:
            you_fucked_up(bearer_token)
        sequence = sequence[1:] + [res]