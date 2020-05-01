import mido
from mido import Message, MidiFile, MidiTrack


class Melody:

    @staticmethod
    def getMidi(self, path):
        return mido.MidiFile(path)

    @staticmethod
    def playMelody(self, path):
        port = mido.open_output()

        mid = self.getMidi(self, path)
        for msg in mid.play():
            port.send(msg)

    @staticmethod
    def midiToDictionary(self, midi):
        notes = []
        times = []
        velocities = []
        types = []
        channels = []

        for i, track in enumerate(midi.tracks):
            for msg in track:
                if(msg.is_meta == False):
                    notes.append(msg.note)
                    times.append(msg.time)
                    velocities.append(msg.velocity)
                    types.append(msg.type)
                    channels.append(msg.channel)

        return {"notes": notes, "times": times, "velocities": velocities, "types": types, "channels": channels,
                "ticks_per_beat": midi.ticks_per_beat}

    @staticmethod
    def dictionaryToMidi(self, dictionary):
        mid = MidiFile(type=0)
        track = MidiTrack()
        mid.tracks.append(track)

        for i in range(len(dictionary["notes"])):
            #todo почему пришлось на 5 домножить время?
            message = mido.Message(dictionary["types"][i], channel=dictionary["channels"][i], note=dictionary["notes"][i], velocity=dictionary["velocities"][i], time=dictionary["times"][i] * 5)
            track.append(message)

        mid.save('../midi/new_song.mid')
        return mid



    @staticmethod
    def convertMelodyToBites(self, midi):
        bytes = []

        for msg in midi.play():
            bytes.append(msg.bytes())

        return bytes

    @staticmethod
    def convertBytesToMelody(self, bytes):
        mido.Message.from_bytes(bytes)

    def __init__(self, midi_path):
        self.path = midi_path
        self.midi = self.midiToDictionary(self, self.getMidi(self, midi_path))
        self.midi_bytes = self.convertMelodyToBites(self, self.getMidi(self, midi_path))