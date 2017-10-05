import tensorflow as tf
import numpy as np

tf.reset_default_graph()
sess = tf.InteractiveSession()

################ Defining functions ################

def OneHotEncoding(lines, no_samples):

        A = np.array([[1.0, 0, 0, 0]])
        C = np.array([[0, 1.0, 0, 0]])
        G = np.array([[0, 0, 1.0, 0]])
        T = np.array([[0, 0, 0, 1.0]])

        All_S_Test = np.zeros((no_samples, 400), dtype=np.float32)

        for k in range(0, no_samples):
            in_seq = lines[k]

            for j in range(0, 100):
                if in_seq[j] == 'A':
                    All_S_Test[k, j * 4:(j + 1) * 4] = A
                elif in_seq[j] == 'C':
                    All_S_Test[k, j * 4:(j + 1) * 4] = C
                elif in_seq[j] == 'G':
                    All_S_Test[k, j * 4:(j + 1) * 4] = G
                else:
                    All_S_Test[k, j * 4:(j + 1) * 4] = T

        return All_S_Test


def conv2d(x, W, b, name = "conv"):
    with tf.name_scope(name):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID') + b


def max_pooling(x, name = "pooling"):
    with tf.name_scope(name):
        return tf.nn.max_pool(x, ksize=[1, 65, 1, 1], strides=[1, 65, 1, 1], padding='VALID') # 135


def fc(x, W, b, name = "fc"):
    with tf.name_scope(name):
        return tf.matmul(x, W) + b


################ One hot encoding ################

Seq = tuple(open("CTCFSeq.txt","r"))

Seq_hotEnc = OneHotEncoding(Seq, len(Seq))


# Creating training and validation set

train_input = Seq_hotEnc[0:77415,:]

train_label = np.concatenate((np.ones((39425,1)), np.zeros((37990,1))), axis = 0)

valid_input = Seq_hotEnc[77415:87415,:]

valid_label = np.concatenate((np.ones((5000,1)), np.zeros((5000,1))), axis = 0)

print train_input.shape, train_label.shape, valid_input.shape, valid_label.shape


################ Initializing weights ################

batch_size = 200

x_ = tf.placeholder(tf.float32, shape=[None, 400]) #680

y_ = tf.placeholder(tf.float32, shape=[None, 1])

W_conv = tf.Variable(tf.random_normal([36,4,1,64]), trainable=True, name = "W")

b_conv = tf.Variable(tf.random_normal([64]), trainable=True, name = "b")

W_fc = tf.Variable(tf.random_normal([64, 1]), trainable=True, name = "W")

b_fc = tf.Variable(tf.random_normal([1]), trainable=True, name = "b")

sess.run(tf.initialize_all_variables())


################ Creating CNN model ################

x = tf.reshape(x_, [-1, 100, 4, 1])

h_conv1 = tf.nn.relu(conv2d(x, W_conv, b_conv, "conv1"))

h_pool1 = max_pooling(h_conv1)

h_pool1_flat = tf.reshape(h_pool1, [-1, 1*1*64])

out = fc(h_pool1_flat, W_fc, b_fc, name = "fc1")

out_sigm = tf.nn.sigmoid(out)


################ Training and evaluation ################

writer = tf.summary.FileWriter("/tmp/TensorflowTutorial/1")

writer.add_graph(sess.graph)

with tf.name_scope("xent"):
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(out, y_), name = 'xent')
    summ_xent = tf.summary.scalar("xent", cross_entropy)

with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.to_float(tf.greater(out_sigm, 0.5)), y_) # tf.equal(tf.argmax(out,1), tf.argmax(y_,1)) #
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    summ_acc = tf.summary.scalar("accuracy", accuracy)

summ = tf.summary.merge([summ_xent, summ_acc]) #tf.summary.merge_all()

for i in range(2001):
    randIndx = np.random.choice(range(77415), batch_size, replace=False)
    batch_xs = train_input[randIndx, :]
    batch_ys = train_label[randIndx, :]
    sess.run(train_step, feed_dict={x_: batch_xs, y_: batch_ys})

    if i % 200 == 0:
        validation = np.zeros((50, 1), dtype=np.float)
        for j in range(50):  # 16
            randIndx_valid = np.arange(j*batch_size, (j+1)*batch_size)
            valid_xs = valid_input[randIndx_valid, :]
            valid_ys = valid_label[randIndx_valid, :]
            summ_, loss_value, accuracy_ = sess.run([summ, cross_entropy, accuracy], feed_dict={x_: valid_xs, y_: valid_ys})
            validation[j] = accuracy_
            writer.add_summary(summ_, i)
        valid_mean = np.mean(validation, axis=0)
        print("step %d, LOSS %g, Accuracy %g" % (i, loss_value, valid_mean[0]))


# To launch Tensorboard from python IDE run the following command. To start from linux terminal exclude '!' at the beginning

# !tensorboard --logdir /tmp/TensorflowTutorial/1


