# -*- coding: utf-8 -*-

import tensorflow as tf
import os
from tqdm import tqdm

from yolo.loss import loss_fn
from yolo.utils.box import visualize_boxes

def train_fn(config_parser, model, train_generator, valid_generator=None, 
        learning_rate=1e-4, epoch=500, save_dname=None, summary_dir=None):
    
    summary_dir = "tmp/summary" if summary_dir is None else summary_dir
    writer = tf.summary.create_file_writer(summary_dir)
    save_fname = _setup(save_dname)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    history = []
    for i in range(epoch):
        with writer.as_default():
            # Train for one epoch
            train_loss = _loop_train(config_parser, model, optimizer, train_generator, i, writer)
            
            # Validate
            if valid_generator:
                valid_loss = _loop_validation(config_parser, model, valid_generator, i, writer)
                loss_value = valid_loss
            else:
                loss_value = train_loss
            
            # Logging onto console after each epoch
            print("{}-th loss = {}, train_loss = {}".format(i, loss_value, train_loss))

            # Write weights file if it is the best one
            history.append(loss_value)
            if save_fname is not None and loss_value == min(history):
                print("    update weight {}".format(loss_value))
                model.save_weights("{}.h5".format(save_fname))
    
    return history


def _loop_train(config_parser, model, optimizer, generator, epoch, writer):
    # one epoch
    
    n_steps = generator.steps_per_epoch
    loss_value = 0
    for i in tqdm(range(n_steps)):
        xs, yolo_1, yolo_2, yolo_3 = generator.next_batch()
        ys = [yolo_1, yolo_2, yolo_3]
        grads, loss = _grad_fn(model, xs, ys)
        loss_value += loss
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if i % 100 == 0:
            step = epoch * n_steps + i
            tf.summary.scalar("training_loss", loss, step=step)
            tf.summary.image("training_image", [xs[0]], step=step)

            image = xs[0] * 255
            detector = config_parser.create_detector(model)
            boxes, labels, probs = detector.detect(image, 0.8)           
            visualize_boxes(image, boxes, labels, probs, config_parser.get_labels())
            tf.summary.image("training_prediction", [image / 255.], step=step)

            writer.flush()

    loss_value /= generator.steps_per_epoch
    return loss_value


def _loop_validation(config_parser, model, generator, epoch, writer):
    # one epoch
    n_steps = generator.steps_per_epoch
    loss_value = 0
    for i in range(n_steps):
        xs, yolo_1, yolo_2, yolo_3 = generator.next_batch()
        ys = [yolo_1, yolo_2, yolo_3]
        ys_ = model(xs)
        loss_value += loss_fn(ys, ys_)        
    loss_value /= generator.steps_per_epoch

    step = (epoch+1) * n_steps
    tf.summary.scalar("validation_loss", loss_value, step=step)
    tf.summary.image("validation_in_image", xs, step=step)

    image = xs[0] * 255
    detector = config_parser.create_detector(model)
    boxes, labels, probs = detector.detect(image, 0.8)           
    visualize_boxes(image, boxes, labels, probs, config_parser.get_labels())
    tf.summary.image("validation_prediction", [image / 255.], step=step)

    writer.flush()

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
