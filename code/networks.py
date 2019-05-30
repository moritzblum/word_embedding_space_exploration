from keras import Sequential
from keras.layers import RepeatVector, LSTM, Dense


# Training 1
def get_LSTM_v1(seq_length, input_dim, output_dim):
    model = Sequential()
    model.add(RepeatVector(seq_length, input_shape=(input_dim,)))
    model.add(LSTM(15, return_sequences=True))  # input_shape=(input_dim, ) not required
    model.add(LSTM(10, return_sequences=True))
    model.add(Dense(70))
    model.add((Dense(output_dim, activation='softmax')))
    return model

# Training 2
def get_LSTM_v2(seq_length, input_dim, output_dim):
    model = Sequential()
    model.add(RepeatVector(seq_length, input_shape=(input_dim,)))
    model.add(LSTM(60, return_sequences=True))  # input_shape=(input_dim, ) not required
    model.add(LSTM(40, return_sequences=True))
    model.add(LSTM(40, return_sequences=True))
    model.add(Dense(70))
    model.add((Dense(output_dim, activation='softmax')))
    return model