import cv2.cv2 as cv2
import json
import numpy as np
import random
import tensorflow as tf

path_base = 'CCTV-Fights/'
path_videos = 'CCTV-Fights/Videos/'

ground_truth = open(path_base + 'groundtruth.json')
data = json.loads(ground_truth.read())
data = data['database']

train, test, validation = [], [], []
train_anomaly, train_normal = [], []
validation_anomaly, validation_normal = [], []
test_anomaly, test_normal = [], []


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


def is_frame_anomaly(index_i, index_j, set):
    res = False
    subset = None
    ranges_frames_anomaly = []

    if set == 'Train':
        subset = train
    if set == 'Validation':
        subset = validation
    if set == 'Test':
        subset = test

    frame_rate = data[subset[index_i]]['frame_rate']
    for dic in data[subset[index_i]]['annotations']:
        segments = dic['segment']
        ranges_frames_anomaly.append([int(frame_rate * segments[0]), int(frame_rate * segments[1])])
    for rang in ranges_frames_anomaly:
        if index_j in range(rang[0], rang[1]):
            res = True

    return res


for clip in data:
    if data[clip]['subset'] == 'training':
        train.append(clip)
    if data[clip]['subset'] == 'testing':
        test.append(clip)
    if data[clip]['subset'] == 'validation':
        validation.append(clip)

for i in range(len(train)):
    video_frames = read_video_optical_flow(path_videos + train[i] + '.mpeg', 20, 20, resize=True)
    for j in range(len(video_frames)):
        fr = video_frames[j]
        if is_frame_anomaly(i, j, set='Train'):
            train_anomaly.append((fr, 1))
        else:
            train_normal.append((fr, 0))

train_total = train_normal + train_anomaly
random.shuffle(train_total)

for i in range(len(validation)):
    video_frames = read_video_optical_flow(path_videos + validation[i] + '.mpeg', 20, 20, resize=True)
    for j in range(len(video_frames)):
        fr = video_frames[j]
        if is_frame_anomaly(i, j, set='Validation'):
            validation_anomaly.append((fr, 1))
        else:
            validation_normal.append((fr, 0))

validation_total = validation_normal + validation_anomaly
random.shuffle(validation_total)

for i in range(len(test)):
    video_frames = read_video_optical_flow(path_videos + test[i] + '.mpeg', 20, 20, resize=True)
    for j in range(len(video_frames)):
        fr = video_frames[j]
        if is_frame_anomaly(i, j, set='Test'):
            test_anomaly.append((fr, 1))
        else:
            test_normal.append((fr, 0))

test_total = test_normal + test_anomaly
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
                frame = cv2.resize(train_total[i][0], (224, 224))
                frame = (frame.astype('float32') - 127.5) / 127.5
                label = train_total[i][1]

                lx.append(frame)
                ly.append(label)

            x = np.array(lx).astype('float32')
            y = np.array(ly).astype('float32')

            x = tf.convert_to_tensor(x)
            y = tf.convert_to_tensor(y)

            yield {'feature': x, 'label': y}


# Validation

def generatorValidationData(batch_size_train=16):
    while True:
        for count in range(int(len(validation_total) / batch_size_train)):
            batch_start = batch_size_train * count
            batch_stop = batch_size_train + (batch_size_train * count)
            lx = []
            ly = []
            for i in range(batch_start, batch_stop):
                frame = cv2.resize(validation_total[i][0], (224, 224))
                frame = (frame.astype('float32') - 127.5) / 127.5
                label = validation_total[i][1]

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
                frame = cv2.resize(test_total[i][0], (224, 224))
                frame = (frame.astype('float32') - 127.5) / 127.5
                label = test_total[i][1]

                lx.append(frame)
                ly.append(label)

            x = np.array(lx).astype('float32')
            y = np.array(ly).astype('float32')

            x = tf.convert_to_tensor(x)
            y = tf.convert_to_tensor(y)

            yield {'feature': x, 'label': y}
