# %pip install pretty_midi
# %pip install libfmp
# basics
from os import defpath
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
# music
import pretty_midi
import IPython.display as ipd
import scipy.io.wavfile
import libfmp.c1

class get_midi:

    def __init__(self, df, filename=None, score=True, audiofile=False):
      self.df = df
      self.filename = filename
      print(f"Mapping file received: {filename}")
      if self.filename:
        self.midiname = f"{filename}.mid"
      else:
        self.midiname = "output.mid"

      midifile = True
      self.export = [midifile, score, audiofile]

    def __build(self):
      midi_data = None
      score = None
      audio = None

      midi_filepath = f"export/music/{self.midiname}" # Construct the full path

      if self.export[0]:
         if self.filename:
          self.__dataframe_to_midi(self.df, output_filename=self.midiname, export=True)
         else:
          self.__dataframe_to_midi(self.df, output_filename=self.midiname, export=True)


      # Load the MIDI file using the full path
      midi_data = pretty_midi.PrettyMIDI(midi_filepath)
      if self.export[1]:

        if self.filename:
            score = self.midi_to_score(midi_data)
            self.visualize_score(score, filename=self.filename)
        else:
            # print("No filename provided. Visualizing score without saving.")
            score = self.midi_to_score(midi_data)
            self.visualize_score(score, filename="score") # Use a default filename for the score
      else:
        score = self.midi_to_score(midi_data)
        self.visualize_score(score, filename="score", export=False)



      if self.export[2]:
        if self.filename:
            self.audio = self.midi_to_audio(midi_data, filename=self.filename, export=True)
            # ipd.Audio(f"{filename}.wav", rate=44100) # Moved display to the calling code if needed
        else:
            self.audio = self.midi_to_audio(midi_data, filename="audio", export=True) # Use a default filename for audio
            # ipd.Audio(f"{filename}.wav", rate=44100) # Moved display to the calling code if needed
      else: # Only synthesize audio if audiofile is False
        self.audio = self.midi_to_audio(midi_data, export=False)
        # ipd.Audio(f"{filename}.wav", rate=44100) # Moved display to the calling code if needed


      return midi_data, score, self.audio


    def build(self):
        """
        Public method to trigger the building process and generate outputs.
        """
        return self.__build()


    def __dataframe_to_midi(self, df, output_filename="output.mid", export=True):
        """
        Converts a pandas DataFrame containing MIDI note information into a MIDI file.

        Args:
            df: pandas DataFrame with columns 'Time (ticks)', 'Duration',
                'Note Number', and 'Velocity'.
            output_filename: The name of the MIDI file to save.
        """
        # Create a PrettyMIDI object
        midi = pretty_midi.PrettyMIDI()

        # Create a an instrument instance for a piano
        piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        piano = pretty_midi.Instrument(program=piano_program)

        # Add notes to the instrument
        for index, row in df.iterrows():
            start_time = row['Time (ticks)'] / midi.resolution
            channel = row['Channel']
            end_time = (row['Time (ticks)'] + row['Duration']) / midi.resolution
            note_number = int(row['Note Number'])
            velocity = int(row['Velocity'])

            # Ensure note number and velocity are within valid MIDI range (0-127)
            note_number = max(0, min(127, note_number))
            velocity = max(0, min(127, velocity))

            note = pretty_midi.Note(velocity=velocity, pitch=note_number, start=start_time, end=end_time)
            piano.notes.append(note)

        # Add the piano instrument to the PrettyMIDI object
        midi.instruments.append(piano)

        if export: # Only export if export is True and filename is provided
          # Write the MIDI file to disk
          midi.write(f"export/music/{output_filename}")
          print(f"MIDI file saved as {output_filename} in the export/music folder")

        # midi_data = pretty_midi.PrettyMIDI(midi)
        # print(midi_data)


        # Always return the midi object
        # return pretty_midi.PrettyMIDI(midi)


    def midi_to_score(self, midi):
        """Convert a midi file to a list of note events

        Notebook: C1/C1S2_MIDI.ipynb

        Args:
            midi (str or pretty_midi.pretty_midi.PrettyMIDI): Either a path to a midi file or PrettyMIDI object

        Returns:
            score (list): A list of note events where each note is specified as
                ``[start, duration, pitch, velocity, label]``
        """

        if isinstance(midi, str):
            midi_data = pretty_midi.PrettyMIDI(midi)
        elif isinstance(midi, pretty_midi.PrettyMIDI):
            midi_data = midi
        else:
            raise RuntimeError('midi must be a path to a midi file or pretty_midi.PrettyMIDI')

        score = []

        for instrument in midi_data.instruments:
            for note in instrument.notes:
                start = note.start
                duration = note.end - start
                pitch = note.pitch
                channel = instrument.program
                velocity = note.velocity / 127.
                score.append([start, duration, pitch, velocity, instrument.name])
        return score

    def visualize_score(self, score, filename="score", export=True):
        libfmp.c1.visualize_piano_roll(score, figsize=(8, 3), velocity_alpha=True);

        if export:
          plt.savefig(f'export/music/{filename}_score.png',dpi=300)
          print(f"Score file saved as {filename}_score.png in the export/music folder")

        plt.show()


    def midi_to_audio(self, midi_data, sr=44100,filename="audio", export=False):
        """
        Converts a PrettyMIDI object or MIDI file path to an audio waveform.

        Args:
            midi_data: A PrettyMIDI object or a string path to a MIDI file.
            sr: The desired sampling rate for the audio.

        Returns:
            A NumPy array representing the audio waveform.
        """
        if isinstance(midi_data, str):
            midi = pretty_midi.PrettyMIDI(midi_data)
        elif isinstance(midi_data, pretty_midi.PrettyMIDI):
            midi = midi_data
        else:
            raise TypeError("Input must be a PrettyMIDI object or a file path string.")

        # Synthesize the MIDI data to a waveform
        # print(f"Sampling rate:{sr}")
        print("""+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+""")
        print("Synthesizing MIDI data to audio preview...")
        audio = midi.synthesize(fs=sr)
        display(ipd.Audio(audio, rate=44100))
        print("""+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+""")

        if export and filename:
            scipy.io.wavfile.write(f"export/music/{filename}.wav", sr, audio)
            print(f"Audio file saved as {filename}.wav in the export/music folder")
        return audio