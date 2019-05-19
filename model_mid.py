import tensorflow as tf

# # IDE mode
# from tensorflow.python import keras
# from tensorflow.python.keras.layers import Dense, GRU, Dropout, Embedding

# running mode
from tensorflow import keras
from tensorflow.keras.layers import Dense, GRU, Dropout, Embedding

import numpy as np
import time
import pickle

embedding_size = 16
epochs = 50
batch_size = 32
id = "_game"
weights_filename = 'output/my_model_weights '+id+'.h5'
songs_data_filename = 'data/game.npy'
start_note = '<start>'
end_note = '<end>'
MAX_LEN = 3000

# my_device = '/GPU:0'
my_device = '/GPU:0'


class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding = Embedding(num_notes, embedding_size)
        self.gru1 = GRU(512, return_sequences=True, return_state=True)
        self.dropout1 = Dropout(0.4)
        self.gru2 = GRU(512, return_sequences=True, return_state=True)
        self.dropout2 = Dropout(0.4)
        self.gru3 = GRU(512, return_sequences=True, return_state=True)
        self.dropout3 = Dropout(0.4)
        self.gru4 = GRU(512, return_sequences=True, return_state=True)
        self.dropout4 = Dropout(0.4)

        self.d = Dense(num_notes, activation='softmax')

    def call(self, x, state=(None, None, None, None)):

        x = self.embedding(x)

        x, output_state1 = self.gru1(x, initial_state=state[0])
        x = self.dropout1(x)

        x, output_state2 = self.gru2(x, initial_state=state[1])
        x = self.dropout2(x)

        # x, output_state3 = self.gru3(x, initial_state=state[2])
        # x = self.dropout3(x)
        #
        # x, output_state4 = self.gru4(x, initial_state=state[3])
        # x = self.dropout4(x)

        x = self.d(x)

        return x, (output_state1, output_state2, output_state3, output_state4)


def loss_function(y, output):
    return scc(y, output, sample_weight=tf.math.not_equal(y, 0))


def np_array_to_music_sentences(songs_raw):

    def notes_to_words(array):

        if np.array_equal([-1., -1., -1., -1., -1., -1.], array):
            return ""

        return "|".join([str(x) for x in array])

    x = np.apply_along_axis(notes_to_words, 2, songs_raw)

    def helper(a):
        a = list(filter(lambda b: b != "", a))
        a = "start " + " ".join(a) + " end"
        return a
    x = np.apply_along_axis(helper, 1, x)

    return x


with open(songs_data_filename, 'rb') as f:
    songs = np.load(f)
    songs_strings = np_array_to_music_sentences(songs)
# tf.debugging.set_log_device_placement(True)

tokenizer = keras.preprocessing.text.Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(songs_strings)
tokenized = tokenizer.texts_to_sequences(songs_strings)
num_notes = len(tokenizer.word_index) + 1

x = keras.preprocessing.sequence.pad_sequences(tokenized, padding='pre', maxlen=MAX_LEN)
n, maxlen = x.shape
y = np.hstack([x[:, 1:], np.zeros((n, 1), dtype=np.int)])

indexes = np.random.permutation(n)
split_index = int(0.9 * n)
x_train = x[indexes[:split_index]]
x_val = x[indexes[split_index:]]

y_train = y[indexes[:split_index]]
y_val = y[indexes[split_index:]]

model = MyModel()

optimizer = keras.optimizers.Adam()
loss_metric_train = keras.metrics.Mean()
loss_metric_val = keras.metrics.Mean()

@tf.function
def train_step(x, y, model):
    with tf.GradientTape() as tape:
        output = model(x)[0]
        loss = loss_function(y, output)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    loss_metric_train(loss)


@tf.function
def val_step(x, y, model):
    output = model(x)[0]
    loss = loss_function(y, output)

    loss_metric_val(loss)

scc = keras.losses.SparseCategoricalCrossentropy()

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(x_val.shape[0])

start = time.time()
metric_result = []
with tf.device(my_device):
    for epoch in range(epochs):
        for x, y in train_data:
            train_res = train_step(x, y, model)

        for x, y in val_data:
            val_res = val_step(x, y, model)
        if epoch % 1 == 0:
            start = time.time()
            metric_result.append([loss_metric_train.result().numpy(), loss_metric_val.result().numpy()])
            end = time.time()
            print("Epoch "+str(epoch)+", Time:"+ str(end-start)+ "s, Loss: "+str(loss_metric_train.result().numpy())+", Validation Loss: "+str(loss_metric_val.result().numpy()))
        loss_metric_train.reset_states()
        loss_metric_val.reset_states()
model.save_weights(weights_filename)
with open("output/training_hist" + id, "wb") as training_hist:
    pickle.dump(metric_result, training_hist)