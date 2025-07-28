import os
import pandas as pd
from get_event import get_event
from get_midi import get_midi

class seismic2midi:
  def __init__(self, eventid,
               filename = None,
               client="IRIS",network='IU',
               station='ANMO', location='00', channel='BHZ',
               location_plot=False,
               fft=False,
               spectrogram=False,
               model= "iasp91",
               fundamental = "C4",
               arrival_rayplot= False,
               #arrival_timeplot = False,
               tempo_bpm = 120,
               score=True, audiofile=False):


    # House Keeping
    if not os.path.exists("../export"):
      os.mkdir("../export")
      print("export folder created")
    if not os.path.exists("../export/seismic"):
      os.mkdir("../export/seismic")
      print("export/seismic folder created")
    if not os.path.exists("../export/music"):
      os.mkdir("../export/music")
      print("export/music folder created")

    if eventid:
      self.eventid=eventid
    else:
      raise ValueError("eventid is required")
    if client:
      self.client=client
    if network:
      self.network=network
    if station:
      self.station=station
    if location:
      self.location=location
    if channel:
      self.channel=channel
    self.model=model
    self.fundamental=fundamental
    self.tempo_bpm=tempo_bpm
    self.export = [location_plot, fft, spectrogram, arrival_rayplot, score, audiofile]

  # eventid="11994157"

  def build(self):
    print("""+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+""")
    print(r"""

                                   ._ o o
                                   \_`-)|_
                                ,""       \
                              ,"  ## |   ಠ ಠ.
                            ," ##   ,-\__    `.
                          ,"       /     `--._;)
                        ,"     ## /
                      ,"   ##    /


                """)
    print(f'Welcone to seismic2midi                              [ようこそ]')
    print("""+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+""")

    epicenter_df, sperical_coordinates_df, arrival_MIDICC_mapping_address = get_event(self.eventid,
              client=self.client,network=self.network,
              station=self.station,location=self.location,
              channel=self.channel, location_plot=self.export[0],
              fft=self.export[1], spectrogram=self.export[2],arrival_rayplot=self.export[3], model=self.model,fundamental=self.fundamental, tempo_bpm=self.tempo_bpm
                                                                                      ).build()
    # print(f"arrival_MIDICC_mapping_address:{arrival_MIDICC_mapping_address}")
    df=pd.read_csv(f'../export/seismic/{arrival_MIDICC_mapping_address}')
    filename=arrival_MIDICC_mapping_address.split('_arrival')[0]
    midi_data, score, audio = get_midi(df=df, filename=filename, score=self.export[4],audiofile=self.export[5]).build()

    print(f""" ////\\
/`O-O'
   ]   We are all set, have fun with the Eventid:{self.eventid} data!
   -    """)
    print("""+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+""")

    # print(f"We are all set, have fun with the Eventid:{self.eventid} data!")

    return epicenter_df, sperical_coordinates_df, midi_data, score, audio