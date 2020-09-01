import re
import io
import os
import music21 as m21
import json
import tensorflow.keras as keras
import numpy as np
import pandas as pd

# google drive imports
from googleapiclient.discovery import build
import pickle
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.http import MediaIoBaseDownload

KERN_DATASET_PATH = "deutschl/test"
ACCEPTABLE_DURATIONS = [
    0.25, 0.5, 0.75, 1.0, 1.5, 2, 3, 4
]
SAVE_DIR = "dataset"
SINGLE_FILE_DATASET = "file_dataset"
SEQUENCE_LENGTH = 64
MAPPING_PATH = 'mapping.json'
MY_MELODIES_CSV_PATH = "my_melodies.csv"
MIDI_MELODIES_PATH = "midi resources\\melodies"

# google drive consts
CLIENT_ID = "837354965689-3cgu6kg68fdg4gijgit7786msc4nt6ba.apps.googleusercontent.com"
CLIENT_SECRET = "MScoYrhOlc2PwYNYlps_GYTN"
CLIENT_SECRET_FILE = 'credentials.json'
API_NAME = "drive"
API_VERSION = 'v3'
SCOPES = ["https://www.googleapis.com/auth/drive"]


def load_songs_in_kern(dataset_path):
    songs = []

    # go through all the files in dataset and load them with music21
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:

            # consider only kern files
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs


def preprocess(dataset_path):
    # load folk songs
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")

    for i, song in enumerate(songs):

        # filter out songs that have non-acceptable durations
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue

        # transpose songs to Cmaj/Amin
        song = transpose(song)

        # encode songs with music time series representation
        encoded_song = encode_song(song)

        # save songs to text file
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)


def transpose(song):
    # get key from the song
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]

    # estimate key using music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    # get interval for transposition. E.g., Bmaj -> Cmaj
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # transpose song by calculated interval
    tranposed_song = song.transpose(interval)
    return tranposed_song


def has_acceptable_durations(song, acceptable_durations):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True


def encode_song(song, time_step=0.25):
    encoded_song = []

    for event in song.flat.notesAndRests:

        # handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi  # 60
        # handle rests
        elif isinstance(event, m21.note.Rest):
            symbol = "r"

        # convert the note/rest into time series notation
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):

            # if it's the first time we see a note/rest, let's encode it. Otherwise, it means we're carrying the same
            # symbol in a new time step
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # cast encoded song to str
    encoded_song = " ".join(map(str, encoded_song))

    return encoded_song


def load_song(file_path):
    with open(file_path, 'r') as fp:
        song = fp.read()

    return song


def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    new_song_delimiter = "/ " * sequence_length
    songs = ""

    # load encoded songs and add delimiters
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load_song(file_path)
            songs = songs + song + " " + new_song_delimiter

    # remove empty space from last character of string
    songs = songs[:-1]

    # save string that contains all the dataset
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)

    return songs


def create_mapping(songs, mapping_path):
    mappings = {}

    # identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))

    # create mappings
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i

    # save voabulary to a json file
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)

    return songs


def convert_songs_to_int(songs):
    int_songs = []

    with open(MAPPING_PATH, 'r') as fp:
        mappings = json.load(fp)

    songs = songs.split()

    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs


def generate_training_sequences(sequence_length):
    # load songs and map them to int
    songs = load_song(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)

    inputs = []
    targets = []

    # generate the training sequences
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i + sequence_length])
        targets.append(int_songs[i + sequence_length])

    # one-hot encode the sequences
    vocabulary_size = len(set(int_songs))
    # inputs size: (# of sequences, sequence length, vocabulary size)
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)

    return inputs, targets


def main():
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)
    print(inputs, targets)


def normalize_my_melodies_csv(csv_path):
    df_uncleaned = pd.read_csv(csv_path)
    df_uncleaned.columns = ["timestamp", "email", "mood", "artist_name", "song_name", "bpm", "key", "instrument",
                            "main_genre", "second_genre", "midi_download_url"]
    df_uncleaned.info()

    file_names = []
    for i in range(len(df_uncleaned['timestamp'])):
        clean_file_name = str(i + 1) + " " + df_uncleaned['artist_name'][i] + " - " + df_uncleaned['song_name'][
            i] + " ({})".format(df_uncleaned['instrument'][i]) + " " + str(df_uncleaned['bpm'][i]) + "bpm " + df_uncleaned['key'][i] + ".mid"
        clean_file_name = re.sub(r'[\\/*?:"<>|]', "", clean_file_name)
        file_names.append(clean_file_name)

    midi_pathes = download_midi_from_google_drive(df_uncleaned['midi_download_url'], file_names)

    df_uncleaned['midi_local_path'] = midi_pathes

    df_uncleaned.to_csv("clean_melodies_data.csv")


def Create_Service(client_secret_file, api_name, api_version, *scopes):
    print(client_secret_file, api_name, api_version, scopes, sep='-')
    CLIENT_SECRET_FILE = client_secret_file
    API_SERVICE_NAME = api_name
    API_VERSION = api_version
    SCOPES = [scope for scope in scopes[0]]
    print(SCOPES)

    cred = None

    pickle_file = f'token_{API_SERVICE_NAME}_{API_VERSION}.pickle'
    # print(pickle_file)

    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as token:
            cred = pickle.load(token)

    if not cred or not cred.valid:
        if cred and cred.expired and cred.refresh_token:
            cred.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            cred = flow.run_local_server()

        with open(pickle_file, 'wb') as token:
            pickle.dump(cred, token)

    try:
        service = build(API_SERVICE_NAME, API_VERSION, credentials=cred)
        print(API_SERVICE_NAME, 'service created successfully')
        return service
    except Exception as e:
        print('Unable to connect.')
        print(e)
        return None


def download_midi_from_google_drive(urls, file_names):
    service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)
    file_ids = [s.split("id=")[1] for s in urls]
    midi_pathes = []

    for file_id, file_name in zip(file_ids, file_names):
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fd=fh, request=request)
        done = False

        while not done:
            status, done = downloader.next_chunk()
            print("Download progress {0}".format(status.progress() * 100))

            fh.seek(0)

            current_file_path = os.path.join(MIDI_MELODIES_PATH, file_name)
            midi_pathes.append(current_file_path)

            with open(current_file_path, 'wb') as f:
                f.write(fh.read())
                f.close()

    return midi_pathes


if __name__ == "__main__":
    normalize_my_melodies_csv(MY_MELODIES_CSV_PATH)
