from preprocess import generate_training_sequences, SEQUENCE_LENGTH, MAPPING_PATH, KERN_DATASET_PATHES, SINGLE_FILE_DATASET
import tensorflow.keras as keras
import json
import os

OUTPUT_UNITS = 34
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
NUM_UNITS = [256]
EPOCHS = 50
BATCH_SIZE = 64
SAVE_MODEL_PATH = 'models/'
INPUT_DATA_PATH = 'output/'
DATASET_NAME = 'file_dataset'
MAPPING_NAME = 'mapping'


def build_model(output_units, num_units, loss, learning_rate):
    input = keras.layers.Input(shape=(None, output_units))
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)

    output = keras.layers.Dense(output_units, activation="softmax")(x)

    model = keras.Model(input, output)

    optimizer = keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    model.summary()

    return model


def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE, single_file_dataset=SINGLE_FILE_DATASET, mapping_path=MAPPING_PATH, dataset_name='model'):
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH, single_file_dataset=single_file_dataset, mapping_path=mapping_path)
    model = build_model(output_units, num_units, loss, learning_rate)
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)
    model.save(SAVE_MODEL_PATH + dataset_name + ".h5")


if __name__ == '__main__':
    kern_dataset_genres = [c.split('/')[-1] for c in KERN_DATASET_PATHES]
    kern_dataset_pathes = [INPUT_DATA_PATH + DATASET_NAME + "_" + c for c in kern_dataset_genres]
    mapping_pathes = [INPUT_DATA_PATH + MAPPING_NAME + "_" + c + ".json" for c in kern_dataset_genres]

    print(kern_dataset_genres)
    print(kern_dataset_pathes)

    for i in range(len(kern_dataset_pathes)):
        with open(mapping_pathes[i], 'r') as fp:
            output_units = len(json.load(fp))
        train(output_units=output_units, single_file_dataset=kern_dataset_pathes[i], mapping_path=mapping_pathes[i], dataset_name=kern_dataset_genres[i])