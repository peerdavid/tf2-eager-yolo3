# -*- coding: utf-8 -*-

import tensorflow as tf
import os
from tqdm import tqdm

from yolo.loss import loss_fn


def train_fn(model, train_generator, valid_generator=None, 
        learning_rate=1e-4, epoch=500, save_dname=None, summary_dir=None):
    
    summary_dir = "tmp/summary" if summary_dir is None else summary_dir
    writer = tf.summary.create_file_writer(summary_dir)
    save_fname = _setup(save_dname)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    history = []
    for i in range(epoch):
        with writer.as_default():
            # Train for one epoch
            train_loss = _loop_train(model, optimizer, train_generator, i)
            
            # Validate
            if valid_generator:
                valid_loss = _loop_validation(model, valid_generator)
                loss_value = valid_loss
            else:
                loss_value = train_loss
            
            # Logging onto console and tensorboard
            print("{}-th loss = {}, train_loss = {}".format(i, loss_value, train_loss))
            tf.summary.scalar("train_loss", train_loss, step=i)
            tf.summary.scalar("validation_loss", loss_value, step=i)

            # Write weights file if it is the best one
            history.append(loss_value)
            if save_fname is not None and loss_value == min(history):
                print("    update weight {}".format(loss_value))
                model.save_weights("{}.h5".format(save_fname))
        
        writer.flush()
    
    return history


def _loop_train(model, optimizer, generator, epoch):
    # one epoch
    
    n_steps = generator.steps_per_epoch
    loss_value = 0
    for _ in tqdm(range(n_steps)):
        xs, yolo_1, yolo_2, yolo_3 = generator.next_batch()
        ys = [yolo_1, yolo_2, yolo_3]
        grads, loss = _grad_fn(model, xs, ys)
        loss_value += loss
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
   
    tf.summary.image("Input", xs, step=epoch)
    loss_value /= generator.steps_per_epoch
    return loss_value


def _loop_validation(model, generator):
    # one epoch
    n_steps = generator.steps_per_epoch
    loss_value = 0
    for _ in range(n_steps):
        xs, yolo_1, yolo_2, yolo_3 = generator.next_batch()
        ys = [yolo_1, yolo_2, yolo_3]
        ys_ = model(xs)
        loss_value += loss_fn(ys, ys_)
    loss_value /= generator.steps_per_epoch
    return loss_value


def _setup(save_dname):
    if save_dname:
        if not os.path.exists(save_dname):
            os.makedirs(save_dname)
        save_fname = os.path.join(save_dname, "weights")
    else:
        save_fname = None
    return save_fname


def _grad_fn(model, images_tensor, list_y_trues):
    with tf.GradientTape() as tape:
        logits = model(images_tensor)
        loss = loss_fn(list_y_trues, logits)
        # print("loss = ", loss)
    return tape.gradient(loss, model.trainable_variables), loss


if __name__ == '__main__':
    pass
