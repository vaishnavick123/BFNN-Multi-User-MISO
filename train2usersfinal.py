import numpy as np
#from utils2_bf import *
#from utils2_keras import *	
#from utils_const_vd import *
#from kaggle_utils import *
from tensorflow.python.keras.layers import *
from matplotlib import pyplot as plt
# tf.config.experimental_run_functions_eagerly(True)
# ------------------------------------------
#  Load and generate simulation data
# ------------------------------------------
#path = 'D:\\work\\3rd year\\6th sem\\capestone\\User_2\\07-05-2021\\data_set_23_04\\0db\\example\\Train'
#path = '../input/user-2-data-17-06/user_2data_17_06/0db/Train'
path = '../input/user-2-data-17-06/user_2data_17_06/-20db/Train'
#path = 'C:/Users/Vaishnavi/Desktop/6 sem project (capstone)/dnn2users_+20db/Train'
#print("\n please enter the batchsize you want to run the model")
#batchsize = int(input("please enter the batchsize you want to run the model"))
#print("the batch size of the model is -->" + str(batchsize))
# Noticed that this is only a default path containing few samples to test if the program can run successfully
# you can download provided train sets or trained weights from the given google driver in readme
H_1, H_1_est, H_2, H_2_est = mat_load(path)
print("\n the shape of the channel estimate of user 1 H_1 is ")
print(H_1.shape)
# here wer are concatinating both the channel estimate data 
R = 100000
#R is the slicing/sample size , for 10^5 samples, R = 100000
H = np.concatenate((H_1,H_2), axis = 2)
#H = H[range(0,R),:]
print("\n the shape of the H")
print(H.shape)
# this is the estimate channel state information of both users  that is used to generate the V_rf 
H_est = np.concatenate((H_1_est,H_2_est), axis = 2)
print("\n the shape of the H_est")
#H_est = H_est[range(0,R),:]
print(H_est.shape)

# use the estimated csi as the input of the BFNN
H_input = np.expand_dims(np.concatenate([np.real(H_est), np.imag(H_est)], 1), 1)
H_input = np.resize(H_input,(R,1,2,128))
print("\n the shape of the H_input")
print(H_input.shape)

# H denotes the perfect csi
H = np.resize(H,(R,128))
H_est = np.resize(H_est,(R,128))
print("\n the shape of the H_squeeze")
print(H.shape)
# generate  SNRs associated with different samples
SNR = np.power(10, np.random.randint(-20, 20, [H.shape[0], 1]) / 10)




# -----------------------
#  Construct the BFNN Model
# -----------------------
# imperfect CSI is used to output the vrf
imperfect_CSI = Input(name='imperfect_CSI', shape=(H_input.shape[1:4]), dtype=tf.float32)
print("\n the shape of the imperfect_CSI")
print(imperfect_CSI.shape)
# perfect_CSI is only used to compute the loss, and not required in prediction
perfect_CSI = Input(name='perfect_CSI', shape=(H.shape[1],), dtype=tf.complex64)
print("\n the shape of the perfect_CSI ")
print(perfect_CSI.shape)
# the SNR is also fed into the BFNN
SNR_input = Input(name='SNR_input', shape=(1,), dtype=tf.float32)
temp = BatchNormalization()(imperfect_CSI)
temp = Flatten()(temp)
temp = BatchNormalization()(temp)
temp = Dense(512, activation='relu')(temp)
temp = BatchNormalization()(temp)
temp = Dense(256, activation='relu')(temp)
phase = Dense(2*Nt)(temp)
V_RF = Lambda(trans_Vrf, dtype=tf.complex64, output_shape=(Nt,))(phase)
rate = Lambda(Rate_func, dtype=tf.float32, output_shape=(1,))([perfect_CSI, V_RF, SNR_input])
#rate = Lambda(Rate_func, dtype=tf.float32, output_shape=(1,))([perfect_CSI, V_RF, SNR_input, batchsize])

model = Model(inputs=[imperfect_CSI, perfect_CSI, SNR_input], outputs=rate)
# the y_pred is the actual rate, thus the loss is y_pred, without labels
model.compile(optimizer='adam', loss=lambda y_true, y_pred: y_pred)
#model.compile(optimizer='adam', loss = )
model.summary()

batch = batchsize
# -----------------------
#  Train Your Model
# -----------------------
#factor 0.2 patience 20 monitor = val_loss
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=0.000005)
checkpoint = tf.keras.callbacks.ModelCheckpoint('./neg20dbbatch1002users_200epochstry1.h5', monitor='val_loss',
                                       verbose=0, save_best_only=True, mode='min', save_weights_only=True)

history = model.fit(x=[H_input, H, SNR], y=H, batch_size=batch,
          epochs=20, verbose=2, validation_split=0.1, callbacks=[reduce_lr, checkpoint])
#-------------------------
#visualize model history
#-------------------------
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# -----------------------
#  Test Your Model
# -----------------------
rate = []
# model.load_weights('./arpita.h5')
for snr in range(-20, 25, 5):
    SNR = np.power(10, np.ones([H.shape[0], 1]) * snr / 10)
    y = model.evaluate(x=[H_input, H, SNR], y=H, batch_size=batch)
    print(snr, y)
    rate.append(-y)
print(rate)

plt.title("The result of BFNN training with -20db")
plt.xlabel("SNR(dB)")
plt.ylabel("Spectral Efficiency")
plt.plot(range(-20, 25, 5), rate)
plt.show()
