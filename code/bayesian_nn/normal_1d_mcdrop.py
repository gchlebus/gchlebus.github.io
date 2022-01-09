__author__ = 'gchlebus'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tqdm import tqdm

def f(x, sigma):
  epsilon = np.random.randn(*x.shape) * sigma
  return 10 * np.sin(2 * np.pi * (x)) + epsilon

train_size = 32
noise = 1.0

X = np.linspace(-0.5, 0.5, train_size).reshape(-1, 1)
y = f(X, sigma=noise)
y_true = f(X, sigma=0.0)

in_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))
out_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))
drop_rate = 0.25
units = 20
input = tf.keras.layers.Input(shape=(1,))
x = tf.keras.layers.Dense(units, activation="relu")(input)
x = tf.keras.layers.Dropout(drop_rate)(x)
x = tf.keras.layers.Dense(units, activation="relu")(x)
#x = tf.keras.layers.Dropout(drop_rate)(x)
x = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(input, x)


batch_size = train_size

y_pred = model(in_placeholder)
loss_op = tf.reduce_mean(tf.pow(y_pred - out_placeholder, 2))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.03)
train_op = optimizer.minimize(loss_op)
init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                     tf.compat.v1.local_variables_initializer())

sess = tf.compat.v1.Session()


sess.run(init_op)
for step in tqdm(range(10000), desc="training"):
  _, loss = sess.run([train_op, loss_op], feed_dict={
    in_placeholder: X,
    out_placeholder: y,
    K.learning_phase(): 1
  })
  if step % 100 == 0:
    print("loss:", loss)

X_test = np.linspace(-1.5, 1.5, 1000).reshape(-1, 1)
y_pred_list = []


for i in tqdm(range(500), desc="inference"):
  y_pre = sess.run(y_pred, feed_dict={
    in_placeholder: X_test,
    K.learning_phase(): 1
  })
  y_pred_list.append(y_pre)

y_preds = np.concatenate(y_pred_list, axis=1)

y_mean = np.mean(y_preds, axis=1)
y_sigma = np.std(y_preds, axis=1)

fig = plt.figure(figsize=(10,8))
plt.plot(X, y_true, label='Truth')
plt.plot(X_test, y_mean, 'r-', label='Predictive mean')
plt.scatter(X, y, marker='+', label='Training data')
plt.fill_between(X_test.ravel(),
                 y_mean + 2 * y_sigma,
                 y_mean - 2 * y_sigma,
                 alpha=0.5, label='Epistemic uncertainty')
plt.title('Prediction')
plt.legend()
fig.tight_layout()
fig.savefig("output_mcdrop_begin_%.1f.pdf" % drop_rate, dpi=300)