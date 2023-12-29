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
test_videos = [l[0] + '.mp4' for l in list(csv.reader(open('UBI_Fights/test_videos.csv')))]
train_videos = [p for p in videos if [p] not in test_videos]

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
for i in range(len(train_videos)):
    if 'F' in train_videos[i]:
        video_frames = read_video_optical_flow(path_videos + 'fight/' + train_videos[i], 224, 224, resize=True)
    else:
        video_frames = read_video_optical_flow(path_videos + 'normal/' + train_videos[i], 224, 224, resize=True)
    frames_label = list(csv.reader(open(path_base + train_videos[i].split('.')[0] + '.csv')))
    for j in range(len(video_frames)):
        fr = video_frames[j]
        label = frames_label[j][0]
        train_total.append((fr, label))
random.shuffle(train_total)


test_total = []
for i in range(len(test_videos)):
    if 'F' in test_videos[i]:
        video_frames = read_video_optical_flow(path_videos + 'fight/' + test_videos[i], 224, 224, resize=True)
    else:
        video_frames = read_video_optical_flow(path_videos + 'normal/' + test_videos[i], 224, 224, resize=True)
    frames_label = list(csv.reader(open(path_base + test_videos[i].split('.')[0] + '.csv')))
    for j in range(len(video_frames)):
        fr = video_frames[j]
        label = frames_label[j][0]
        test_total.append((fr, label))
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
