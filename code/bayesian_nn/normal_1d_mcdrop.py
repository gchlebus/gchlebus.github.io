__author__ = 'gchlebus'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
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

input = tf.keras.layers.Input(shape=(1,))
x = tfp.layers.DenseFlipout(20, activation="relu")(input)
x = tfp.layers.DenseFlipout(20, activation="relu")(x)
x = tfp.layers.DenseFlipout(1)(x)
model = tf.keras.Model(input, x)


batch_size = train_size
num_batches = train_size / batch_size
kl_weight = 1.0 / train_size
print("kl_weight", kl_weight)

y_pred = model(in_placeholder)
y_dist = tfp.distributions.Normal(loc=y_pred,scale=noise)
log_likelihood = y_dist.log_prob(out_placeholder)
neg_log_likelihood = -tf.reduce_mean(log_likelihood)
kl = sum(model.losses) * kl_weight
elbo_loss = neg_log_likelihood + kl

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.03)
train_op = optimizer.minimize(elbo_loss)
init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                     tf.compat.v1.local_variables_initializer())

sess = tf.compat.v1.Session()


sess.run(init_op)
for step in tqdm(range(10000), desc="training"):
  _, loss = sess.run([train_op, elbo_loss], feed_dict={
    in_placeholder: X,
    out_placeholder: y
  })

X_test = np.linspace(-1.5, 1.5, 1000).reshape(-1, 1)
y_pred_list = []


for i in tqdm(range(500), desc="inference"):
  y_pre = sess.run(y_pred, feed_dict=
    {in_placeholder: X_test})
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
fig.savefig("output_kl_%.2f.pdf" % kl_weight, dpi=300)