import os
import requests
import mido

from src.classes.Melody import Melody


def playMelody(path):
    port = mido.open_output()

    mid = mido.MidiFile(path)
    for msg in mid.play():
        port.send(msg)


def downloadFile(url):
    response = requests.get(url, stream=True)

    text_file = open("./midi/data.sqlite", "wb")
    for chunk in response.iter_content(chunk_size=1024):
        text_file.write(chunk)

    text_file.close()


def midiToArray(midi):
    notes = []
    times = []
    velocities = []
    types = []

    for msg in midi.play():
        notes.append(msg.note)
        times.append(msg.time)
        velocities.append(msg.velocity)
        types.append(msg.type)

    return {"notes": notes, "times": times, "velocities": velocities, "types": types}


melody = Melody("../midi/test2.mid")
print(melody.midi)
midi2 = melody.dictionaryToMidi(melody, melody.midi)

# mid = mido.MidiFile("../midi/test2.mid")
# print(midiToArray(mid))