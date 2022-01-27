import cv2.cv2 as cv2
import csv
import numpy as np
import random
import tensorflow as tf
import os
from sklearn.model_selection import StratifiedShuffleSplit

path_base = 'UBI_Fights/annotation/'
path_videos = 'UBI_Fights/videos/'

annotations = os.listdir(path_base)
videos = os.listdir(path_videos + 'fight/') + os.listdir(path_videos + 'normal/')
label_videos = list()

for v in videos:
    if 'F' in v:
        label_videos.append(1)
    else:
        label_videos.append(0)

width = 224
height = 224
channels = 3


def read_video_optical_flow(vid, width, height, resize=False):
    video_frames_optical_flow = list()
    i = 0
    cap = cv2.VideoCapture(vid)
    ret1, frame1 = cap.read()
    if resize:
        frame1 = cv2.resize(frame1, (width, height))
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    if not cap.isOpened():
        print("Error opening video stream or file")

    while cap.isOpened():
        ret2, frame2 = cap.read()
        if ret2:
            if resize:
                frame2 = cv2.resize(frame2, (width, height))
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            bgr = np.reshape(bgr, (width, height, channels))
            video_frames_optical_flow.append(bgr)
        else:
            break
        i += 1
        prvs = next
    cap.release()
    cv2.destroyAllWindows()
    return video_frames_optical_flow


def read_video(vid, width, height, resize=False):
    video_frames = list()
    cap = cv2.VideoCapture(vid)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if resize:
                frame = cv2.resize(frame, (width, height))
                frame = np.reshape(frame, (width, height, channels))
            video_frames.append(frame)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    return video_frames


X_train = []
y_train = []
X_valid_test = []
y_valid_test = []
X_test = []
y_test = []
X_valid = []
y_valid = []

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
for train_index, test_valid_index in split.split(videos, label_videos):
    for ti in train_index:
        X_train.append(videos[ti])
        y_train.append(label_videos[ti])

    for tsi in test_valid_index:
        X_valid_test.append(videos[tsi])
        y_valid_test.append(label_videos[tsi])

split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
for test_index, valid_index in split2.split(X_valid_test, y_valid_test):
    for tssi in test_index:
        X_test.append(X_valid_test[tssi])
        y_test.append(y_valid_test[tssi])

    for tvi in valid_index:
        X_valid.append(X_valid_test[tvi])
        y_valid.append(y_valid_test[tvi])


train_total = []
for i in range(len(X_train)):
    if 'F' in X_train[i]:
        video_frames = read_video_optical_flow(path_videos + 'fight/' + X_train[i], 20, 20, resize=True)
    else:
        video_frames = read_video_optical_flow(path_videos + 'normal/' + X_train[i], 20, 20, resize=True)
    frames_label = list(csv.reader(open(path_base + X_train[i].split('.')[0] + '.csv')))
    for j in range(len(video_frames)):
        fr = video_frames[j]
        label = frames_label[j][0]
        train_total.append((fr, label))
random.shuffle(train_total)


validation_total = []
for i in range(len(X_valid)):
    if 'F' in X_valid[i]:
        video_frames = read_video_optical_flow(path_videos + 'fight/' + X_valid[i], 20, 20, resize=True)
    else:
        video_frames = read_video_optical_flow(path_videos + 'normal/' + X_valid[i], 20, 20, resize=True)
    frames_label = list(csv.reader(open(path_base + X_valid[i].split('.')[0] + '.csv')))
    for j in range(len(video_frames)):
        fr = video_frames[j]
        label = frames_label[j][0]
        validation_total.append((fr, label))
random.shuffle(validation_total)


test_total = []
for i in range(len(X_test)):
    if 'F' in X_test[i]:
        video_frames = read_video_optical_flow(path_videos + 'fight/' + X_test[i], 20, 20, resize=True)
    else:
        video_frames = read_video_optical_flow(path_videos + 'normal/' + X_test[i], 20, 20, resize=True)
    frames_label = list(csv.reader(open(path_base + X_test[i].split('.')[0] + '.csv')))
    for j in range(len(video_frames)):
        fr = video_frames[j]
        label = frames_label[j][0]
        test_total.append((fr, label))
random.shuffle(test_total)


# Train

def generatorTrainData(batch_size_train=16):
    while True:
        for tp in train_total:
            frame = cv2.resize(tp[0], (width, height), interpolation=cv2.INTER_AREA)
            frame = 127.5 - (frame.astype('float32') / 127.5)
            label = tp[1]
            for count in range(int(len(train_total) / batch_size_train)):
                batch_start = batch_size_train * count
                batch_stop = batch_size_train + (batch_size_train * count)
                lx = []
                ly = []
                for i in range(batch_start, batch_stop):
                    lx.append(frame)
                    ly.append(label)

                lx = np.array(lx).astype('float32')
                ly = np.array(ly).astype('float32')

                x = tf.convert_to_tensor(lx)
                y = tf.convert_to_tensor(ly)
                yield {'feature': x, 'label': y}


# Validation

def generatorValidationData(batch_size_train=16):
    while True:
        for tp in validation_total:
            frame = cv2.resize(tp[0], (width, height), interpolation=cv2.INTER_AREA)
            frame = 127.5 - (frame.astype('float32') / 127.5)
            label = tp[1]
            for count in range(int(len(validation_total) / batch_size_train)):
                batch_start = batch_size_train * count
                batch_stop = batch_size_train + (batch_size_train * count)
                lx = []
                ly = []
                for i in range(batch_start, batch_stop):
                    lx.append(frame)
                    ly.append(label)

                lx = np.array(lx).astype('float32')
                ly = np.array(ly).astype('float32')

                x = tf.convert_to_tensor(lx)
                y = tf.convert_to_tensor(ly)
                yield {'feature': x, 'label': y}


# Test

def generatorTestData(batch_size_test=16):
    while True:
        for tp in test_total:
            frame = cv2.resize(tp[0], (width, height), interpolation=cv2.INTER_AREA)
            frame = 127.5 - (frame.astype('float32') / 127.5)
            label = tp[1]
            for count in range(int(len(test_total) / batch_size_test)):
                batch_start = batch_size_test * count
                batch_stop = batch_size_test + (batch_size_test * count)
                lx = []
                ly = []
                for i in range(batch_start, batch_stop):
                    lx.append(frame)
                    ly.append(label)

                lx = np.array(lx).astype('float32')
                ly = np.array(ly).astype('float32')

                x = tf.convert_to_tensor(lx)
                y = tf.convert_to_tensor(ly)
                yield {'feature': x, 'label': y}
