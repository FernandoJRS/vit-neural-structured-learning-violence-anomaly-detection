import neural_structured_learning as nsl
import time
import tensorflow_hub as hub
import datetime
from Data_CCTV import *


loaded_model = hub.load("HubModels/vit_s16_fe_1")
vit_model = tf.keras.Sequential(
    [tf.keras.layers.InputLayer(input_shape=(width, height, channels)),
     hub.KerasLayer(loaded_model, trainable=True),
     tf.keras.layers.Dense(16),
     tf.keras.layers.Dense(2, activation='softmax')])

adv_config = nsl.configs.make_adv_reg_config(multiplier=0.2,
                                             adv_step_size=0.05,
                                             adv_grad_norm='infinity')

adv_model = nsl.keras.AdversarialRegularization(vit_model,
                                                label_keys=['label'],
                                                adv_config=adv_config)

tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True)

adv_model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

log_dir = "CCTV-Fights/Results/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint_path = "CCTV-Fights/Results/logs/checkpoint/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

start_time_train = time.time()
initial_weights = adv_model.get_weights()
adv_model.fit(generatorTrainData(batch_size_train=16),
              epochs=5,
              steps_per_epoch=int(len(train_total) / 16),
              validation_data=generatorValidationData(batch_size_train=16),
              validation_steps=int(len(validation_total) / 16),
              callbacks=[tensorboard_callback, cp_callback])
adv_model.set_weights(initial_weights)
print('Training time per epoch: ' + str((time.time() - start_time_train) / 5))

start_time_test = time.time()
adv_model.evaluate(generatorTestData(batch_size_test=16),
                   steps=int(len(test_total) / 16))
print('Inference time: ' + str((time.time() - start_time_test) / len(test_total)))
