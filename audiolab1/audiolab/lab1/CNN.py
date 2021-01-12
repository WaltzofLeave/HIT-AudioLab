import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from make_dataset import get_dataset
from get import get
import wave
# import soundfile as sf

def plotresult(wave_data:np.ndarray,label:list):
    assert wave_data.shape[0] == len(label)
    x = list(range(0, wave_data.shape[0] * wave_data.shape[1]))
    x = np.array(x)
    x = np.reshape(x, (-1, 256))
    for i in range(0, wave_data.shape[0]):
        if label[i] == 0:
            plt.plot(x[i],wave_data[i],color='blue')
        else:
            plt.plot(x[i],wave_data[i],color='red')
    plt.show()

class Loader():
    def __init__(self,dataset,sepnum):
        self.training_data,self.training_label = dataset[0:sepnum,0:256],dataset[0:sepnum,256:257]
        self.validation_data,self.validation_label = dataset[sepnum:,0:256],dataset[sepnum:,256:257]
        self.training_data = np.expand_dims(self.training_data,axis=-1)
        self.validation_data = np.expand_dims(self.validation_data,axis=-1)
        self.training_label = np.expand_dims(self.training_label,axis=-1)
        self.validation_label = np.expand_dims(self.validation_label,axis=-1)
        self.training_data_amount = np.shape(self.training_data)[0]
        self.validation_data_amount = np.shape(self.validation_data)[0]
    def get_batch(self,batch_size):
        index = np.random.randint(0,np.shape(self.training_data)[0],batch_size)
        return self.training_data[index,:],self.training_label[index]
class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(
            filters=16,
            kernel_size=5,
            activation=tf.keras.activations.relu,
            padding="valid"
        )
        self.conv2 = tf.keras.layers.Conv1D(
            filters=16,
            kernel_size=5,
            activation=tf.keras.activations.relu,
            padding="valid"
        )
        self.maxpool1 = tf.keras.layers.MaxPool1D(
            pool_size=2
        )
        self.dropout1 = tf.keras.layers.Dropout(rate=0.1)
        self.conv3 = tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=3,
            activation=tf.keras.activations.relu,
            padding='valid'
        )
        self.conv4 = tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=3,
            activation=tf.keras.activations.relu,
            padding='valid'
        )
        self.maxpool2 = tf.keras.layers.MaxPool1D(pool_size=2)
        self.dropout2 = tf.keras.layers.Dropout(rate=0.1)
        self.conv5 = tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=3,
            activation=tf.keras.activations.relu,
            padding='valid'
        )
        self.conv6 = tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=3,
            activation=tf.keras.activations.relu,
            padding='valid'
        )
        self.maxpool3 = tf.keras.layers.MaxPool1D(pool_size=2)
        self.dropout3 = tf.keras.layers.Dropout(rate=0.1)
        self.conv7 = tf.keras.layers.Conv1D(
            filters=256,
            kernel_size=3,
            activation=tf.keras.activations.relu,
            padding='valid'
        )
        self.conv8 = tf.keras.layers.Conv1D(
            filters=256,
            kernel_size=3,
            activation=tf.keras.activations.relu,
            padding='valid'
        )
        self.maxpool4 = tf.keras.layers.GlobalMaxPool1D()
        self.dropout4 = tf.keras.layers.Dropout(rate=0.2)
        self.dense1 = tf.keras.layers.Dense(
            units=64,
            activation=tf.keras.activations.relu
        )
        self.dense2 = tf.keras.layers.Dense(
            units=64,
            activation=tf.keras.activations.relu
        )
        self.dense3 = tf.keras.layers.Dense(
            units=2,
            activation=tf.keras.activations.softmax
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.maxpool4(x)
        x = self.dropout4(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x
def train(plotting=True):
    epoch_amount = 10
    batch_size = 50
    gamma = 0.001
    model = CNN()
    data_loader = Loader(get_dataset(),2500)
    my_optimizer = tf.keras.optimizers.Adam(learning_rate=gamma)
    loss_func = tf.keras.losses.sparse_categorical_crossentropy
    total_iterations = int(data_loader.training_data_amount //
                       batch_size*epoch_amount)

    print(total_iterations, 'iterations needed,Proceed?')
    for batch in range(total_iterations):
        pass
        Input, Validation = data_loader.get_batch(batch_size=batch_size)
        with tf.GradientTape() as tape:
            Prediction = model(Input)
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                y_true=Validation, y_pred=Prediction)

            loss = tf.reduce_mean(loss)
        print("batch %d: loss %f" % (batch, loss.numpy()))
        grads = tape.gradient(loss, model.variables)
        my_optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    acc_func = tf.keras.metrics.SparseCategoricalAccuracy()
    total_iterations = int(
        data_loader.validation_data_amount//batch_size*epoch_amount)
    
    model.summary()
    if plotting:
        for i in range(1,11):
            ansi = []
            wave_data = get(i)
            for frame in wave_data:
                frame1 = np.expand_dims(frame, axis=0)
                frame1 = np.expand_dims(frame1, axis=-1)
                result = model.predict(frame1)
                if result[0][1] >= 0.5:
                    ansi.append(1)
                else:
                    ansi.append(0)
            plotresult(wave_data,ansi)
    for i in range(1,11):
        ansi = None
        wave_data = get(i)
        for frame in wave_data:
            #print("Frame:")
            #print(frame)
            #print(frame.shape)
            frame1 = np.expand_dims(frame,axis=0)
            frame1 = np.expand_dims(frame1,axis=-1)
            #print('+++++++frame1++++++\n',frame1.shape)
            result =  model.predict(frame1)
            #print(result)
            if result[0][1] >= 0.5:

                if ansi is None:
                    ansi = frame
                    ansi = np.expand_dims(ansi,axis=0)
                else:
                    new = np.expand_dims(frame,axis=0)
                    ansi = np.vstack((ansi,new))
        with wave.open('../result/lab1_2/'+str(i)+'.wav','wb') as f:
            data_to_write = ansi.flatten()
            data_to_write = np.array(data_to_write,dtype=np.dtype(np.int16))

            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.setcomptype('NONE','Uncompressed')

            f.writeframes(data_to_write)


    for batch in range(total_iterations):
        start_index = batch*batch_size
        end_index = (batch+1)*batch_size
        y_pred = model.predict(data_loader.validation_data[start_index:end_index])
        acc = acc_func(y_true=data_loader.validation_label[start_index:end_index], y_pred=y_pred)
        print(acc)

trainning = True
if trainning:
    train()
