
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys
import tensorflow as tf

from base import MLP, BatchManager

def train_multiclass(x, y, num_classes,
          encoder_shape = [100, 100, 2], decoder_shape = [2, 100, 100], learner_shape = [2, 100, 100], recon_weight = 5,
          learning_rate = 0.001, batch_size = 4, min_epochs = 100, stopping_epochs = 50, tol = 0.001, freq_eval = 1):
    
    # Allow multiple sessions on a single GPU.
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # Reset TF graph (avoids issues with repeat experiments)
    tf.compat.v1.reset_default_graph()
    
    # Setup directory
    os.system("rm -rf Model")
    cwd = os.getcwd()
    os.makedirs("Model")
    os.chdir("Model")
    
    sys.stdout = open("train.txt", "w")
    
    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size = 0.5)
    
    # Get sizes for future reference
    n = x_train.shape[0]
    n_input = x_train.shape[1]
    encoder_shape.insert(0, n_input)
    decoder_shape.append(n_input)
    learner_shape.append(num_classes)

    # Batch Manager
    bm = BatchManager(x_train, y_train)

    # Graph inputs
    X = tf.compat.v1.placeholder("float", [None, n_input], name = "X_in")
    D = tf.compat.v1.placeholder(tf.float32, shape=[1, n_input])
    R = tf.compat.v1.placeholder("float", [None, 2], name = "R_in")
    Y = tf.compat.v1.placeholder(tf.int64, [None], name = "Y_in")
    I = tf.compat.v1.placeholder(tf.int64, shape = None)

    # Build the models
    encoder = MLP(encoder_shape)
    with tf.compat.v1.variable_scope("encoder_model", reuse = tf.compat.v1.AUTO_REUSE):
        rep = encoder.model(X + D)
    
    decoder = MLP(decoder_shape)
    with tf.compat.v1.variable_scope("decoder_model", reuse = tf.compat.v1.AUTO_REUSE):
        recon = decoder.model(rep)

    learner = MLP(learner_shape)
    with tf.compat.v1.variable_scope("learner_model", reuse = tf.compat.v1.AUTO_REUSE):
        pred = learner.model(rep)
        prob = tf.nn.softmax(pred, axis = 1)
        prob_i = tf.gather(prob, indices = [I], axis = 1)
        grad_i = tf.gradients(ys=prob_i, xs=X)

    with tf.compat.v1.variable_scope("learner_model", reuse = tf.compat.v1.AUTO_REUSE):
        pred_from_rep = learner.model(R)

    # Define the loss and optimizer
    model_loss = tf.reduce_mean(input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(labels = Y, logits = pred))
    tf.compat.v1.summary.scalar("Cross Entropy", model_loss)
    
    accuracy = tf.reduce_mean(input_tensor=tf.cast(tf.equal(tf.argmax(input=pred, axis = 1), Y), tf.float32))
    tf.compat.v1.summary.scalar("Accuracy", accuracy)

    recon_loss = tf.compat.v1.losses.mean_squared_error(labels = X, predictions = recon)
    tf.compat.v1.summary.scalar("Recon MSE", recon_loss)
    
    loss_op = model_loss + recon_weight * recon_loss
    tf.compat.v1.summary.scalar("Loss", loss_op)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(loss_op)

    summary_op = tf.compat.v1.summary.merge_all()

    # Train and evaluate the model
    init = [tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()]
    saver = tf.compat.v1.train.Saver(max_to_keep = 1)

    best_epoch = 0
    best_loss = np.inf

    sess = tf.compat.v1.Session(config = tf_config)

    train_writer = tf.compat.v1.summary.FileWriter("train", sess.graph)
    val_writer = tf.compat.v1.summary.FileWriter("val")

    sess.run(init)
    epoch = 0
    total_batch = int(n / batch_size)
    while True:

        # Stopping condition
        if epoch - best_epoch > stopping_epochs and epoch > min_epochs:
            break

        # Run a training epoch
        for i in range(total_batch):
            x_batch, y_batch = bm.next_batch(batch_size = batch_size)
            summary, _ = sess.run([summary_op, train_op], feed_dict = {X: x_batch, Y: y_batch, D: np.zeros((1, n_input))})
            train_writer.add_summary(summary, epoch * total_batch + i)

        # Run model metrics
        if epoch % freq_eval == 0:
            
            summary, val_loss = sess.run([summary_op, loss_op], feed_dict = {X: x_val, Y: y_val, D: np.zeros((1, n_input))})
            
            if val_loss < best_loss - tol:
                print(epoch, " ", val_loss)
                best_loss = val_loss
                best_epoch = epoch
                saver.save(sess, "./model.cpkt")
    
            val_writer.add_summary(summary, (epoch + 1) * total_batch)

        epoch += 1

    train_writer.close()
    val_writer.close()

    # Evaluate the final model
    saver.restore(sess, "./model.cpkt")
    print("Test Accuracy: ", sess.run(accuracy, {X: x_test, Y: y_test, D: np.zeros((1, n_input))}))

    # Find the 2d point representation
    points = sess.run(rep, {X: x, D: np.zeros((1, n_input))})

    plt.scatter(points[:, 0], points[:, 1], s = 10)

    # Plot the function over that space
    '''
    min_0 = np.min(points[:, 0])
    max_0 = np.max(points[:, 0])
    min_1 = np.min(points[:, 1])
    max_1 = np.max(points[:, 1])

    feature_0 = np.linspace(min_0, max_0, 50)
    feature_1 = np.linspace(min_1, max_1, 50)
    r = np.zeros((1, 2))
    map = np.empty((50, 50))
    for i in range(50):
        r[0, 1] = feature_1[i]
        for j in range(50):
            r[0, 0] = feature_0[j]
            map[i, j] = sess.run(pred_from_rep, {R: r})

    plt.contour(feature_0, feature_1, map)
    plt.colorbar()
    '''

    plt.savefig("learned_function.pdf")
    plt.close()

    pickle.dump(points, open("points.pkl", "wb"))

    # Go back to directory
    os.chdir(cwd)
    
    return sess, rep, prob, grad_i, X, D, Y, I
