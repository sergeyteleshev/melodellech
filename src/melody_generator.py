import os
import tensorflow.keras as keras
import json
from preprocess import SEQUENCE_LENGTH, KERN_DATASET_PATHES
from train import SAVE_MODEL_PATH, INPUT_DATA_PATH, MAPPING_NAME
import numpy as np
import music21 as m21
import random

MAPPING_PATH = 'output/mapping_erk.json'
MELODIES_PATH = 'output/midi/'
MODEL_PATH = 'models/erk.h5'
SEED_MIN_LENGTH = 2
SEED_MAX_LENGTH = 15


def generate_random_seed(mapping_path=MAPPING_PATH):
    seed = ""
    seed_length = random.randint(SEED_MIN_LENGTH, SEED_MAX_LENGTH)
    keys = []
    is_pause = False

    with open(mapping_path, 'r') as json_file:
        data = json.load(json_file)
        if "/" in data:
            del data['/']

        keys = list(data.keys())

    for i in range(seed_length):
        if is_pause:
            seed += "_ " * random.randint(1, 8)
            is_pause = True
        else:
            seed += str(random.choice(keys)) + " "
            is_pause = False

    return seed


class MelodyGenerator:

    def __init__(self, model_path=MODEL_PATH, mapping_path=MAPPING_PATH):
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)
        self.mapping_path = mapping_path

        with open(mapping_path, 'r') as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH

    @staticmethod
    def _sample_with_temperature(probabilities, temperature):
        predictions = np.log(probabilities) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))
        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities)

        return index

    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        """Generates a melody using the DL model and returns a midi file.
        :param seed (str): Melody seed with the notation used to encode the dataset
        :param num_steps (int): Number of steps to be generated
        :param max_sequence_len (int): Max number of steps in seed to be considered for generation
        :param temperature (float): Float in interval [0, 1]. Numbers closer to 0 make the model more deterministic.
            A number closer to 1 makes the generation more unpredictable.
        :return melody (list of str): List with symbols representing a melody
        """

        # create seed with start symbols
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        # map seed to int
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):

            # limit the seed to max_sequence_length
            seed = seed[-max_sequence_length:]

            # one-hot encode the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            # (1, max_sequence_length, num of symbols in the vocabulary)
            onehot_seed = onehot_seed[np.newaxis, ...]

            # make a prediction
            probabilities = self.model.predict(onehot_seed)[0]
            # [0.1, 0.2, 0.1, 0.6] -> 1
            output_int = self._sample_with_temperature(probabilities, temperature)

            # update seed
            seed.append(output_int)

            # map int to our encoding
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            # check whether we're at the end of a melody
            if output_symbol == "/":
                break

            # update melody
            melody.append(output_symbol)

        return melody

    @staticmethod
    def save_melody(melody, step_duration=0.25, format="midi", file_name="mel.mid"):
        """Converts a melody into a MIDI file
        :param melody (list of str):
        :param min_duration (float): Duration of each time step in quarter length
        :param file_name (str): Name of midi file
        :return:
        """

        # create a music21 stream
        stream = m21.stream.Stream()

        start_symbol = None
        step_counter = 1

        # parse all the symbols in the melody and create note/rest objects
        for i, symbol in enumerate(melody):

            # handle case in which we have a note/rest
            if symbol != "_" or i + 1 == len(melody):

                # ensure we're dealing with note/rest beyond the first one
                if start_symbol is not None:

                    quarter_length_duration = step_duration * step_counter  # 0.25 * 4 = 1

                    # handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    # handle note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(m21_event)

                    # reset the step counter
                    step_counter = 1

                start_symbol = symbol

            # handle case in which we have a prolongation sign "_"
            else:
                step_counter += 1

        # write the m21 stream to a midi file
        stream.write(format, file_name)


def generate_melodies(num_melodies, mg):
    for i in range(num_melodies):
        seed = generate_random_seed(mapping_path=mg.mapping_path)
        melody = mg.generate_melody(seed, 64, SEQUENCE_LENGTH, 0.7)
        mg_name = mg.model_path.split('/')[-1][:-3]
        mg_save_path = MELODIES_PATH + mg_name

        if not os.path.exists(mg_save_path):
            os.mkdir(mg_save_path)

        mg.save_melody(melody, file_name=mg_save_path + "/" + str(i + 1) + ".mid")
        print("generated melody #" + str(i + 1) + ", using {} model".format(mg_name))


if __name__ == '__main__':
    models_path = [SAVE_MODEL_PATH + c.split('/')[-1] + ".h5" for c in KERN_DATASET_PATHES]
    kern_dataset_genres = [c.split('/')[-1] for c in KERN_DATASET_PATHES]
    mapping_pathes = [INPUT_DATA_PATH + MAPPING_NAME + "_" + c + ".json" for c in kern_dataset_genres]
    models = [MelodyGenerator(model_path=p, mapping_path=mp) for p, mp in zip(models_path, mapping_pathes)]

    for model in models:
        generate_melodies(10, model)

    # mg = MelodyGenerator(model_path='models/allerkbd.h5', mapping_path='output/mapping_allerkbd.json')
    # seed = "67 _ _ _ 65 _ _ 69 _ _"
    # # seed2 = "67 _ _ _ _ _ 65 _ 64 _ 62 _ 60 _ _ _"
    # # seed = mg.generate_random_seed()
    # melody = mg.generate_melody(seed, 64, SEQUENCE_LENGTH, 0.7)
    # print(melody)
    # mg.save_melody(melody, file_name="metluha_loh.mid")
    # generate_melodies(10)
