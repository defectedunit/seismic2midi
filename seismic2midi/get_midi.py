# %pip install pretty_midi
# %pip install libfmp
# basics
from os import defpath
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import math
# music
import pretty_midi
import IPython.display as ipd
import scipy.io.wavfile
import pypianoroll
# import libfmp.c1



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
            # self.visualize_score(score, filename=self.filename)
            self.score2roll(midi_filepath, filename=self.filename,export=True)
        else:
            # print("No filename provided. Visualizing score without saving.")
            score = self.midi_to_score(midi_data)
            self.score2roll(midi_filepath, filename="score",export=True) # Use a default filename for the score
      else:
        score = self.midi_to_score(midi_data)
        # self.visualize_score(score, filename="score", export=False)
        self.score2roll(midi_filepath,export=False)



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

    # def visualize_score(self, score, filename="score", export=True):
        libfmp.c1.visualize_piano_roll(score, figsize=(8, 3), velocity_alpha=True);

        if export:
          plt.savefig(f'export/music/{filename}_score.png',dpi=300)
          print(f"Score file saved as {filename}_score.png in the export/music folder")

        plt.show()

    def score2roll(self,midi_file_path, filename="score",export=False):
  
        multitrack = pypianoroll.read(midi_file_path)

        # multitrack = pypianoroll.from_pretty_midi(midi_data)

        track = multitrack.tracks[0]

        # --- Create a custom colormap with white background
        original_cmap = cm.get_cmap('turbo')

        # Create a new colormap where the first color (for 0 velocity) is white
        colors = original_cmap(np.arange(original_cmap.N)) # Get all colors from the original cmap
        colors[0] = np.array([1, 1, 1, 1])  # Set the first color (for 0) to white (RGBA)
        custom_cmap = mcolors.ListedColormap(colors)

        # Plot the single track pianoroll with the custom colormap

        fig, ax = plt.subplots(figsize=(9, 2))

        # Plot the single track pianoroll with the custom colormap on the created axes
        track.plot(ax=ax, cmap=custom_cmap) 

        # --- Set y-axis limits from C0 to C8 and set yticks ---
        ax.set_ylim(12, 127) # Set limits to cover 0-127 MIDI notes
        ax.set_yticks(np.arange(0, 128, 20)) # Set ticks every 12 MIDI notes (octaves)
        ax.set_yticklabels(np.arange(0, 128, 20)) # Label these ticks with their MIDI numbers
        ax.set_ylabel('Pitch') # Explicitly set y-axis label


        # --- Calculate and set x-axis for time in seconds ---
        total_ticks = track.pianoroll.shape[0]
        resolution = multitrack.resolution  # Ticks per beat

        # Get tempo. multitrack.tempo can be a scalar or an array.
        # For simplicity, use the first tempo value if it's an array, or the scalar directly.
        if isinstance(multitrack.tempo, np.ndarray):
            # If tempo is an array (tempo changes), we take the first value for an approximation
            tempo_bpm = multitrack.tempo[0]
        else:
            tempo_bpm = multitrack.tempo # beats per minute

        if tempo_bpm == 0: # Avoid division by zero if tempo is somehow 0
            tempo_bpm = 120 # Default to 120 BPM

        beats_per_second = tempo_bpm / 60
        ticks_per_second = resolution * beats_per_second
        total_seconds = total_ticks / ticks_per_second

        # Determine a reasonable interval for x-axis ticks (e.g., every 10 seconds)
        tick_interval_seconds = 10
        tick_interval_ticks = tick_interval_seconds * ticks_per_second

        # Generate tick locations in terms of ticks (for ax.set_xticks)
        x_tick_locations = np.arange(0, total_ticks, tick_interval_ticks)

        # Generate corresponding labels in seconds (for ax.set_xticklabels)
        x_tick_labels = [f'{int(s)}' for s in np.arange(0, total_seconds, tick_interval_seconds)]

        # Set the ticks and labels
        ax.set_xticks(x_tick_locations)
        ax.set_xticklabels(x_tick_labels)
        ax.set_xlabel('Time (s)') # Explicitly set x-axis label
        ax.set_ylim(20, 110)

        ax.set_title(f'{filename}')
        
        if export:
            plt.savefig(f'export/music/{filename}_score.png',dpi=300) # Corrected DPI to dpi
            print(f"Score file saved. as {filename}_score.png")

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