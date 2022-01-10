import neural_structured_learning as nsl
import time
import tensorflow_hub as hub
from Data_CCTV import *

loaded_model = hub.load("HubModels/vit_s16_fe_1")
model = tf.keras.Sequential(
    [hub.KerasLayer(loaded_model, trainable=False),
     tf.keras.layers.Dense(16),
     tf.keras.layers.Dense(2, activation='softmax')])

model.build((None, width, height, channels))
model.summary()

adv_config = nsl.configs.make_adv_reg_config(multiplier=0.2,
                                             adv_step_size=0.05,
                                             adv_grad_norm='infinity')

adv_model = nsl.keras.AdversarialRegularization(model,
                                                label_keys=['label'],
                                                adv_config=adv_config)

adv_model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(),
                           tf.keras.metrics.CategoricalAccuracy()])

log_dir = "CCTV-Fights/Results/logs/fit/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint_path = "CCTV-Fights/Results/logs/checkpoint/"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

start_time_train = time.time()

adv_model.fit(generatorTrainData(batch_size_train=256),
              epochs=1,
              steps_per_epoch=int(len(train_total) / 256),
              validation_data=generatorValidationData(batch_size_train=256),
              validation_steps=int(len(validation_total) / 256),
              callbacks=[tensorboard_callback, cp_callback])
print('Training time per epoch: ' + str((time.time() - start_time_train) / 256))

adv_model.evaluate(generatorTestData(batch_size_test=256),
                   steps=int(len(test_total) / 256))