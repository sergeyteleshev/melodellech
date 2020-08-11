import os
import music21 as m21

KERN_DATASET_PATH = "deutschl/test"
ACCEPTABLE_DURATIONS = [
    0.25, 0.5, 0.75, 1.0, 1.5, 2, 3, 4
]

def load_songs_in_kern(dataset_path):
    songs = []

    for path, subdirs, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == 'krn':
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)

    return songs


def preprocess(dataset_path):
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")

    for song in songs:
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue

        song = transpose(song)


def transpose(song):
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]

    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    if key.mode == 'major':
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == 'minor':
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    transposed_song = song.transpose(interval)

    return transposed_song


def has_acceptable_durations(song, acceptable_durations):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False

    return True


if __name__ == "__main__":
    songs = load_songs_in_kern(KERN_DATASET_PATH)
    print(f"Loaded {len(songs)} songs.")
    song = songs[0]
    t = transpose(song)
    print(has_acceptable_durations(song, ACCEPTABLE_DURATIONS))
    print(has_acceptable_durations(t, ACCEPTABLE_DURATIONS))
    print(song.analyze("key"))
    print(t.analyze("key"))