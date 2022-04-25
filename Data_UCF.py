import cv2.cv2 as cv2
import numpy as np
import random
import tensorflow as tf
import os

path_videos = 'UCF_Crimes/Videos/'
path_splits = 'UCF_Crimes/Action_Recognition_splits/'
videos = os.listdir(path_videos)


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
train_read = os.listdir(path_splits + 'train_001.txt')

for i in range(len(train_read)):
    video_frames = read_video_optical_flow(path_videos + train_read[i], 20, 20, resize=True)
    for j in range(len(video_frames)):
        fr = video_frames[j]
        if 'Abuse' in train_read[i]:
            train_total.append((fr, 0))
        if 'Arrest' in train_read[i]:
            train_total.append((fr, 1))
        if 'Arson' in train_read[i]:
            train_total.append((fr, 2))
        if 'Assault' in train_read[i]:
            train_total.append((fr, 3))
        if 'Burglary' in train_read[i]:
            train_total.append((fr, 4))
        if 'Explosion' in train_read[i]:
            train_total.append((fr, 5))
        if 'Fighting' in train_read[i]:
            train_total.append((fr, 6))
        if 'Normal_Videos_event' in train_read[i]:
            train_total.append((fr, 7))
        if 'RoadAccidents' in train_read[i]:
            train_total.append((fr, 8))
        if 'Robbery' in train_read[i]:
            train_total.append((fr, 9))
        if 'Shooting' in train_read[i]:
            train_total.append((fr, 10))
        if 'Shoplifting' in train_read[i]:
            train_total.append((fr, 11))
        if 'Stealing' in train_read[i]:
            train_total.append((fr, 12))
        if 'Vandalism' in train_read[i]:
            train_total.append((fr, 13))

random.shuffle(train_total)

test_total = []
test_read = os.listdir(path_splits + 'test_001.txt')

for i in range(len(test_read)):
    video_frames = read_video_optical_flow(path_videos + test_read[i], 20, 20, resize=True)
    for j in range(len(video_frames)):
        fr = video_frames[j]
        if 'Abuse' in test_read[i]:
            test_total.append((fr, 0))
        if 'Arrest' in test_read[i]:
            test_total.append((fr, 1))
        if 'Arson' in test_read[i]:
            test_total.append((fr, 2))
        if 'Assault' in test_read[i]:
            test_total.append((fr, 3))
        if 'Burglary' in test_read[i]:
            test_total.append((fr, 4))
        if 'Explosion' in test_read[i]:
            test_total.append((fr, 5))
        if 'Fighting' in test_read[i]:
            test_total.append((fr, 6))
        if 'Normal_Videos_event' in test_read[i]:
            test_total.append((fr, 7))
        if 'RoadAccidents' in test_read[i]:
            test_total.append((fr, 8))
        if 'Robbery' in test_read[i]:
            test_total.append((fr, 9))
        if 'Shooting' in test_read[i]:
            test_total.append((fr, 10))
        if 'Shoplifting' in test_read[i]:
            test_total.append((fr, 11))
        if 'Stealing' in test_read[i]:
            test_total.append((fr, 12))
        if 'Vandalism' in test_read[i]:
            test_total.append((fr, 13))

random.shuffle(test_total)


# Train

def generatorTrainData(batch_size_train=16):
    while True:
        for count in range(int(len(train_total) / batch_size_train)):
            batch_start = batch_size_train * count
            batch_stop = batch_size_train + (batch_size_train * count)
            lx = []
            ly = []
            for i in range(batch_start, batch_stop):
                frame = cv2.resize(train_total[i][0], (width, height))
                frame = (frame.astype('float32') - 127.5) / 127.5
                label = train_total[i][1]

                lx.append(frame)
                ly.append(label)

            x = np.array(lx).astype('float32')
            y = np.array(ly).astype('float32')

            x = tf.convert_to_tensor(x)
            y = tf.convert_to_tensor(y)

            yield {'feature': x, 'label': y}


# Test

def generatorTestData(batch_size_test=16):
    while True:
        for count in range(int(len(test_total) / batch_size_test)):
            batch_start = batch_size_test * count
            batch_stop = batch_size_test + (batch_size_test * count)
            lx = []
            ly = []
            for i in range(batch_start, batch_stop):
                frame = cv2.resize(test_total[i][0], (width, height))
                frame = (frame.astype('float32') - 127.5) / 127.5
                label = test_total[i][1]

                lx.append(frame)
                ly.append(label)

            x = np.array(lx).astype('float32')
            y = np.array(ly).astype('float32')

            x = tf.convert_to_tensor(x)
            y = tf.convert_to_tensor(y)

            yield {'feature': x, 'label': y}
