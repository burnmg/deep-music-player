import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, GRU, Dropout, Embedding
from music21 import converter, instrument, note, chord
from google.colab import drive
import numpy as np
import time
import glob
import pickle

embedding_size = 8
start_note = 'start'
end_note = 'end'
songs_data_filename = 'data/game.npy'
tokenizer_filename = 'tokenizer_game'
weights_filename = 'my_model_weights.h5'
new_songs_filename = 'new_songs_game100'

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

with open(tokenizer_filename, 'rb') as f:
  tokenizer = pickle.load(f)
num_notes = len(tokenizer.word_index) + 1

x = np.zeros((2, 100))
# x = keras.preprocessing.sequence.pad_sequences(tokenized, padding='post')
n, maxlen = x.shape
y = np.hstack([x[:, 1:], np.zeros((n, 1), dtype=np.int)])

n = 2
indexes = np.random.permutation(n)
split_index = int(0.9 * n)
x_train = x[indexes[:split_index]]
x_val = x[indexes[split_index:]]
y_train = y[indexes[:split_index]]
y_val = y[indexes[split_index:]]


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

    def call(self, x, state=(None, None, None)):

        x = self.embedding(x)

        x, output_state1 = self.gru1(x, initial_state=state[0])
        x = self.dropout1(x)

        x, output_state2 = self.gru2(x, initial_state=state[1])
        x = self.dropout2(x)

        x, output_state3 = self.gru3(x, initial_state=state[2])
        x = self.dropout3(x)
        #
        # x, output_state4 = self.gru4(x, initial_state=state[3])
        # x = self.dropout4(x)

        x = self.d(x)

        return x, (output_state1, output_state2, output_state3)
model = MyModel()

optimizer = keras.optimizers.Adam()
loss_metric_train = keras.metrics.Mean()
loss_metric_val = keras.metrics.Mean()

scc = keras.losses.SparseCategoricalCrossentropy()
def loss_function(y, output):
  return scc(y, output, sample_weight=tf.math.not_equal(y, 0))


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

epochs = 1
batch_size = 1

# model.compile(optimizer='adam', loss=loss_function)
# model.fit(x, y, epochs=epochs, validation_split=0.1)

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(x_val.shape[0])

start = time.time()
for epoch in range(epochs):
  for x, y in train_data:
    train_step(x, y, model)
  for x, y in val_data:
    val_step(x, y, model)
  if epoch % 1 == 0:
    end = time.time()
    print(f'Epoch {epoch}, Time: {end - start: .3f}s, Loss: {loss_metric_train.result(): .4f}, Validation Loss: {loss_metric_val.result(): .4f}')
    start = time.time()

model.load_weights(weights_filename)

def generate(model, k=20):
  result = [tokenizer.word_index[start_note]]
  state = (None, None, None, None)
  while True:
    seq = np.array([[result[-1]]])
    output, state = model(seq, state=state)
    note = np.random.choice(np.argpartition(output[0][0], -k)[-k:])
    if note == 0:
      result.append(tokenizer.word_index[end_note])
    else:
      result.append(note)
    if result[-1] == tokenizer.word_index[end_note] or len(result) >= maxlen:
      break
  return result[1:-1]

generated_data = [generate(model) for i in range(100)]

new_songs = [[tokenizer.index_word[i] for i in g] for g in generated_data]

with open(new_songs_filename, 'wb') as f:
  pickle.dump(new_songs, f)

