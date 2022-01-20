import cv2.cv2 as cv2
import numpy as np
import random
import tensorflow as tf
import os

path_base = 'XD-Violence/'
path_videos_train = 'XD-Violence/Train/'
path_videos_test = 'XD-Violence/Test/'

videos_train = os.listdir(path_videos_train)
videos_test = os.listdir(path_videos_test)

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


train_total = []
for i in range(len(videos_train)):
    video_frames = read_video_optical_flow(path_videos_train + videos_train[i], 20, 20, resize=True)
    for j in range(len(video_frames)):
        fr = video_frames[j]
        if 'A' in videos_train[i]:
            train_total.append((fr, 0))
        if 'B1' in videos_train[i]:
            train_total.append((fr, 1))
        if 'B2' in videos_train[i]:
            train_total.append((fr, 2))
        if 'B4' in videos_train[i]:
            train_total.append((fr, 3))
        if 'B5' in videos_train[i]:
            train_total.append((fr, 4))
        if 'B6' in videos_train[i]:
            train_total.append((fr, 5))
        if 'G' in videos_train[i]:
            train_total.append((fr, 6))
random.shuffle(train_total)


test_total = []
for i in range(len(videos_test)):
    video_frames = read_video_optical_flow(path_videos_test + videos_test[i], 20, 20, resize=True)

    for j in range(len(video_frames)):
        fr = video_frames[j]
        if 'A' in videos_test[i]:
            test_total.append((fr, 0))
        if 'B1' in videos_test[i]:
            test_total.append((fr, 1))
        if 'B2' in videos_test[i]:
            test_total.append((fr, 2))
        if 'B4' in videos_test[i]:
            test_total.append((fr, 3))
        if 'B5' in videos_test[i]:
            test_total.append((fr, 4))
        if 'B6' in videos_test[i]:
            test_total.append((fr, 5))
        if 'G' in videos_test[i]:
            test_total.append((fr, 6))
random.shuffle(test_total)


# Train

def generatorTrainData(batch_size_train=16):
    while True:
        for tp in train_total:
            frame = cv2.resize(tp[0], (width, height), interpolation=cv2.INTER_AREA)
            frame = frame.astype('float32') / 255
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


# Test

def generatorTestData(batch_size_test=16):
    while True:
        for tp in test_total:
            frame = cv2.resize(tp[0], (width, height), interpolation=cv2.INTER_AREA)
            frame = frame.astype('float32') / 255
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
