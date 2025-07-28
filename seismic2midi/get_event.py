# !pip install cartopy
# !pip install obspy
# %pip install pretty_midi

# basics
from os import defpath
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

#seismic
import cartopy.crs as ccrs
import obspy
from obspy.signal.tf_misfit import plot_tfr
from obspy import UTCDateTime
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees
from obspy.clients.fdsn import Client

# music
import pretty_midi



class get_event:
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
               arrival_timeplot = False,
              #  note_map = 1,
               tempo_bpm = 120):
    if eventid:
      self.eventid = eventid
    else:
      raise ValueError("eventid is required")
    if filename:
      self.filename = filename
    else:
      self.filename = f'{eventid}_{network}_{station}_{channel}'

    self.client = Client(client)
    self.network = network
    self.station = station
    self.location = location
    self.channel = channel
    self.filename = filename
    self.model = TauPyModel(model=model)
    self.fundamental = fundamental
    self.tempo_bpm = tempo_bpm

    # if note_map==1:
    #   self.note_map=self.note_map_1()

    self.export = [location_plot, fft, spectrogram,
                   arrival_rayplot, arrival_timeplot]

  def build(self):
    print(f"Fetching Event ID: {self.eventid} form {self.client.base_url}")
    epicenter_df, sperical_coordinates_df, arrival_MIDICC_mapping_address = self.__build()
    return epicenter_df, sperical_coordinates_df, arrival_MIDICC_mapping_address


  def __build(self):
    event_summary = None
    fft = None
    spectrogram = None
    arrival_MIDICC_mapping = None
    arrival_rayplot = None
    arrival_timeplot = None
    spherical_coordinates = None
    st = None

    epicenter_df, sperical_coordinates_df, self.epicenter_lat, self.epicenter_lon, self.epicenter_depth = self.spherical_coordinates()
    st = self.event_trace(self.eventid)

    if self.export[0]:
      fft = self.location_plot(self.eventid)
    if self.export[1]:
      self.event_fft(st)
    if self.export[2]:
      self.event_spectrogram(st)

    self.arrival, phase_arrival,arrival_MIDICC_mapping_address = self.arrival_MIDICC_mapping()
    if self.export[3]:
      self.arrival_rayplot()
    # if self.export[4]:
      # self.arrival_timeplot()


    print(f'Finished get_event for {self.eventid}')

    return epicenter_df, sperical_coordinates_df, arrival_MIDICC_mapping_address



  def get_epicenter_magnitude(self, eventid):
    return self.client.get_events(eventid=eventid)[0]['magnitudes'][0]['mag']

  def get_epicenter(self, eventid):
    epicenter_lat, epicenter_lon, epicenter_depth = self.client.get_events(eventid=eventid)[0]['origins'][0]['longitude'], self.client.get_events(eventid=eventid)[0]['origins'][0]['latitude'], self.client.get_events(eventid=eventid)[0]['origins'][0]['depth']
    event_descriptions = self.client.get_events(eventid=eventid)[0]['event_descriptions'][0]['text']
    event_time = self.client.get_events(eventid=eventid)[0].origins[0].time
    epicenter_mag = self.get_epicenter_magnitude(eventid)
    epicenter_magtype=self.client.get_events(eventid=eventid)[0]['magnitudes'][0]['magnitude_type']
    # print(f'Summary:\neventid:{eventid} is received, magnitude:{epicenter_mag}{epicenter_magtype}, \nOn {event_time}, {event_descriptions}')
    # Construct DataFrame using a dictionary
    epicenter_data = {
        'Summary': ['Eventid', 'Time', 'Location', 'Magnitude', 'Latitude', 'Longitude', 'Depth'],
        'Description': [str(eventid), str(event_time), str(event_descriptions), str(f'{epicenter_mag}{epicenter_magtype}'), str(epicenter_lat), str(epicenter_lon), str(epicenter_depth/1000)]
    }
    epicenter_df = pd.DataFrame(epicenter_data)

    display(epicenter_df)
    return epicenter_df, epicenter_lat, epicenter_lon, epicenter_depth

  def geographic_to_spherical(self, latitude_deg, longitude_deg, depth_km):
    #Convert latitude and longitude from degrees to radians
    latitude_rad = math.radians(latitude_deg)
    longitude_rad = math.radians(longitude_deg)
    depth_km = depth_km/1000
    radius_km = depth_km
    polar_angle_rad = math.pi / 2 - latitude_rad
    azimuthal_angle_rad = longitude_rad

    # print(f"Hypocenter geographical coordinates: Lat={latitude_deg}° N, Lon={longitude_deg}° W, Depth={depth_km} km")
    # print(f"Transformed spherical coordinates: Radius={radius_km:.2f} km, Polar Angle={math.degrees(polar_angle_rad):.2f}° (from Z-axis), Azimuthal Angle={math.degrees(azimuthal_angle_rad):.2f}° (from X-axis)")

    print(rf"""             .-.
            /   \           Hypocenter geographical coordinates:
        ____\___/              Lat={latitude_deg:.2f}° N, Lon={longitude_deg:.2f}° W, Depth={depth_km} km
        \   /\              Transformed spherical coordinates:
           /  \____            Radius={radius_km:.2f} km, Polar Angle={math.degrees(polar_angle_rad):.2f}°, Azimuthal Angle={math.degrees(azimuthal_angle_rad):.2f}°
          |\
          | \                                         __       ((()
         /   /                                  -=   /  \      ///\
________/___/________________________________-=______\__/______\\\/__""")
    print("- " * 35)
    print("__" * 33)

    return radius_km, polar_angle_rad, azimuthal_angle_rad

  def spherical_coordinates(self):
    epicenter_df, epicenter_lat, epicenter_lon, epicenter_depth = self.get_epicenter(self.eventid)
    epic_r, epic_theta, epic_polar = self.geographic_to_spherical(epicenter_lat, epicenter_lon, epicenter_depth)

    sperical_coordinates = [epic_r, epic_theta, epic_polar]
    sperical_coordinates_df = pd.DataFrame(sperical_coordinates, index=['Radius', 'Polar Angle', 'Azimuthal Angle'], columns=['Value'])


    # saving df
    sperical_coordinates_df.to_parquet(f"../export/music/{self.eventid}_spherical_coordinates.parquet")
    print(f"Spherical coordinates saved as {self.eventid}_spherical_coordinates.parquet in the export/music folder")
    epicenter_df.to_parquet(f"../export/seismic/{eventid}_event_summary.parquet")
    print(f"Event summary saved as {eventid}_event_summary.parquet in the export/seismic folder")
    print("+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+")
    

    return epicenter_df, sperical_coordinates_df, epicenter_lat, epicenter_lon, epicenter_depth

  def event_time(self, eventid, export = False):
    return self.client.get_events(eventid=self.eventid)[0].origins[0].time

  def event_trace(self, eventid):
    st = self.client.get_waveforms(network=self.network,station=self.station,
                                   location=self.location, channel=self.channel,starttime=self.event_time(self.eventid),endtime=self.event_time(self.eventid)+3600)

    st[0]
    print(st[0].stats)
    sampling_rate = int(st[0].stats.sampling_rate)
    # st[0].write(f'{st[0].stats.network}-{st[0].stats.station}-{st[0].stats.channel}.wav', format='WAV', framerate=sampling_rate)
    # st[0].write(f'{st[0].stats.network}-{st[0].stats.station}-{st[0].stats.channel}.mseed', format='MSEED')
    return st


  # plotting matters
  def location_plot(self, eventid):
    self.client.get_events(eventid=self.eventid).plot()
    plt.savefig(f'../export/seismic/{self.eventid}_location_plot.png',dpi=300)
    print(f"Event plot saved as {self.eventid}_location_plot.png in the export/seismic folder")

  def event_fft(self, st):
      st[0].plot().savefig(f'../export/seismic/{self.eventid}-{st[0].stats.network}-{st[0].stats.station}-{st[0].stats.channel}_fft.png',dpi=300)
      print(f"FFT saved as {self.eventid}-{st[0].stats.network}-{st[0].stats.station}-{st[0].stats.channel}_fft.png in the export/seismic folder")

  def event_spectrogram(self,st):
      spectrogram = st[0].spectrogram(log=True,
                      title=f'{self.eventid}-{st[0].stats.network}-{st[0].stats.station}-{st[0].stats.channel} ' + str(st[0].stats.starttime),
                      )
      plt.savefig(f'../export/seismic/{self.eventid}-{st[0].stats.network}-{st[0].stats.station}-{st[0].stats.channel}_spectrogram.png',dpi=300)
      print(f"Spectrogram saved as {self.eventid}-{st[0].stats.network}-{st[0].stats.station}-{st[0].stats.channel}_spectrogram.png in the export/seismic folder")

  def note_map(self, phase_arrival):
    #print(f"Arguments received: {args}")
    #print(f"Keyword arguments received: {kwargs}")
    """
    Calculates a score based on the seismic phase name.
    Args:
        phase_arrival: A string representing the seismic phase name.
    Returns:
        An integer representing the calculated score.
    """
    score = 0

    if phase_arrival.startswith("S"):
      score -= 4

    # surface waves inbounds back
    if phase_arrival.startswith("p") or phase_arrival.startswith("s"):
      score += 6

    if phase_arrival.count("diff") != 0:
      oddball = np.random.randint(-12, 13)
      score += oddball
      print(f"Houston, we have a problem, {phase_arrival} is diffracted at CMB, oddball {oddball} is added to the score for {phase_arrival}")

    score += (phase_arrival.count("c")) * 1 # reflected by the outer core
    score += (phase_arrival.count("S")) * (-3) # relative minor
    score += (phase_arrival.count("I")) * 7
    score += (phase_arrival.count("i")) * 5
    score += (phase_arrival.count("K")) * 7

    if phase_arrival.endswith("S"):
      score = -(score)

    return score

  def f_zero(self,f_zero="C4"):
  # Dictionary mapping musical notes (A0 to C8) to MIDI note numbers
    note_to_midi_number = {
        'C0': 12, 'C#0': 13, 'D0': 14, 'D#0': 15, 'E0': 16, 'F0': 17, 'F#0': 18, 'G0': 19, 'G#0': 20, 'A0': 21, 'A#0': 22, 'B0': 23,
        'C1': 24, 'C#1': 25, 'D1': 26, 'D#1': 27, 'E1': 28, 'F1': 29, 'F#1': 30, 'G1': 31, 'G#1': 32, 'A1': 33, 'A#1': 34, 'B1': 35,
        'C2': 36, 'C#2': 37, 'D2': 38, 'D#2': 39, 'E2': 40, 'F2': 41, 'F#2': 42, 'G2': 43, 'G#2': 44, 'A2': 45, 'A#2': 46, 'B2': 47,
        'C3': 48, 'C#3': 49, 'D3': 50, 'D#3': 51, 'E3': 52, 'F3': 53, 'F#3': 54, 'G3': 55, 'G#3': 56, 'A3': 57, 'A#3': 58, 'B3': 59,
        'C4': 60, 'C#4': 61, 'D4': 62, 'D#4': 63, 'E4': 64, 'F4': 65, 'F#4': 66, 'G4': 67, 'G#4': 68, 'A4': 69, 'A#4': 70, 'B4': 71,
        'C5': 72, 'C#5': 73, 'D5': 74, 'D#5': 75, 'E5': 76, 'F5': 77, 'F#5': 78, 'G5': 79, 'G#5': 80, 'A5': 81, 'A#5': 82, 'B5': 83,
        'C6': 84, 'C#6': 85, 'D6': 86, 'D#6': 87, 'E6': 88, 'F6': 89, 'F#6': 90, 'G6': 91, 'G#6': 92, 'A6': 93, 'A#6': 94, 'B6': 95,
        'C7': 96, 'C#7': 97, 'D7': 98, 'D#7': 99, 'E7': 100, 'F7': 101, 'F#7': 102, 'G7': 103, 'G#7': 104, 'A7': 105, 'A#7': 106, 'B7': 107,
        'C8': 108
    }

    return note_to_midi_number[f_zero]

  def arrival_MIDICC_mapping(self, export=False):
      print("-" * 30)
      print(f'Calculating arrival time of Eventid: {self.eventid} at {self.network}.{self.station}..{self.channel}, it may take a while...')
      print("-" * 30)

      inv = self.client.get_stations(network=self.network,station=self.station,level='response')
      coords=inv.get_coordinates(f'{self.network}.{self.station}..{self.channel}')
      distance=locations2degrees(self.epicenter_lat,self.epicenter_lon,coords['latitude'],coords['longitude'])

      arrival=self.model.get_ray_paths(source_depth_in_km=
                                       self.epicenter_depth/1000,distance_in_degree=distance)
      # print(arrival)
      arrival_data = []
      for arr in arrival:
          arrival_data.append({
              'Phase': arr.phase.name,
              'Arrival Time (s)': arr.time,
              'Distance (degrees)': arr.distance,
              'Ray Param': arr.ray_param,
              'Take-off Angle (deg from vertical)': arr.takeoff_angle,
              'Incident Angle (deg from vertical)': arr.incident_angle
          })

      # Create a pandas DataFrame from the extracted data
      arrival_df = pd.DataFrame(arrival_data)
      arrival_df['Note on Arrival']=(arrival_df['Arrival Time (s)']/60).astype(int)
      # Display the DataFrame
      # display(arrival_df[['Phase','Ray Param', 'Arrival Time (s)','Note on Arrival']])

      arrival_df['Score'] = arrival_df['Phase'].apply(self.note_map)
      arrival_df['Note Number'] = arrival_df['Score'] + self.f_zero(f_zero=self.fundamental)
      arrival_df['Time (s)']=(arrival_df['Arrival Time (s)']/60).apply(lambda x: f"{x:.2f}")

      # temp Amplitude place holder
      arrival_df['Velocity']= np.random.randint(20,128, size=len(arrival_df))

      # display(arrival_df[['Phase', 'Arrival Time (s)', 'Score', 'Note Number', 'Velocity', 'Time (s)']])

      ticks_per_beat = pretty_midi.PrettyMIDI().resolution # Default is 220 ticks per beat for pretty_midi
      time_signature_numerator = 4
      time_signature_denominator = 4

      ticks_per_min = (self.tempo_bpm*ticks_per_beat)
      ticks_per_second = ticks_per_min / 60

      # print(f'{ticks_per_beat}')
      # print(f'{ticks_per_second}')
      # Convert time in seconds to time in ticks
      # Time in seconds * (beats per second) * (ticks per beat)
      # beats per second = tempo_bpm / 60
      arrival_df['Arrival Time (min)'] = arrival_df['Arrival Time (s)']/60
      arrival_df['Time (ticks)'] = ((arrival_df['Arrival Time (min)']/2) * ticks_per_second).astype(int)
      arrival_df['Duration'] = (((arrival_df['Ray Param']/60)/30)* ticks_per_second).astype(int)
      arrival_df['Channel'] = arrival_df['Note Number'].apply(lambda x: 1 if x >= arrival_df['Note Number'][0] else 2)


      # Write mapping file
      midi_df=arrival_df[['Time (ticks)', 'Duration', 'Note Number', 'Channel', 'Velocity']] # clean up
      midi_df.to_csv(f'../export/seismic/{self.eventid}_{self.network}_{self.station}_{self.channel}_arrival_MIDICC_mapping.csv', index=False)
      print(f"Mapping data saved as {self.eventid}_{self.network}_{self.station}_{self.channel}_arrival_MIDICC_mapping.csv in the export/seismic folder")
      arrival_MIDICC_mapping_address= f'{self.eventid}_{self.network}_{self.station}_{self.channel}_arrival_MIDICC_mapping.csv'
      # midi_df.to_parquet(f'{self.eventid}_{self.network}_{self.station}_{self.channel}_arrival_MIDICC_mapping.parquet', index=False)
      # print(f" Mapping data saved as {self.eventid}_{self.network}_{self.station}_{self.channel}_arrival_MIDICC_mapping.parqet")
      # arrival_MIDICC_mapping_address= f'{self.eventid}_{self.network}_{self.station}_{self.channel}_arrival_MIDICC_mapping.parqet'


      arrival_MIDICC_mapping_df = arrival_df[['Phase', 'Score', 'Arrival Time (min)', 'Ray Param', 'Time (ticks)', 'Duration',  'Note Number', 'Channel', 'Velocity']]
      print("""\n----------------8<-------------[ The Arrival MIDICC mapping has arrived ]------------------""")
      display(arrival_MIDICC_mapping_df)
      print("""+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
""")
      print("The current mapping schema is not perfect, other schema are in the works.")

      self.phase_arrivals = arrival_df['Phase']
      print("-"*40)
      print(f"# of phase arrivals:{self.phase_arrivals.shape}")
      print("-"*40)

      return arrival, self.phase_arrivals, arrival_MIDICC_mapping_address

  def arrival_rayplot(self):
    fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    # Plot all pre-determined phases
    # Iterate through the Arrival objects in the Arrivals object
    for arr in self.arrival:
    # Access the phase name and distance from each Arrival object
      phase_name = arr.phase.name
      distance_in_degree = arr.distance

      # Now use these values to get and plot the ray paths
      arrivals_to_plot = self.model.get_ray_paths(source_depth_in_km=self.epicenter_depth/1000,
                                            distance_in_degree=distance_in_degree,
                                            phase_list=[phase_name])
      ax = arrivals_to_plot.plot_rays(plot_type='spherical',
                              legend=True, label_arrivals=False,
                              plot_all=True,
                              show=False, ax=ax, indicate_wave_type=True)

    # Annotate regions
    ax.text(0, 0, 'Solid\ninner\ncore',
            horizontalalignment='center', verticalalignment='center',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    ocr = (self.model.model.radius_of_planet -
          (self.model.model.s_mod.v_mod.iocb_depth +
            self.model.model.s_mod.v_mod.cmb_depth) / 2)
    ax.text(np.deg2rad(180), ocr, 'Fluid outer core',
            horizontalalignment='center',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    mr = self.model.model.radius_of_planet - self.model.model.s_mod.v_mod.cmb_depth / 2
    ax.text(np.deg2rad(180), mr, 'Solid mantle',
            horizontalalignment='center',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    plt.savefig(f'../export/seismic/{self.eventid}-{self.network}-{self.station}-{self.channel}_arrival_rayplot.png',dpi=300)
    print(f"Ray plot saved as {self.eventid}-{self.network}-{self.station}-{self.channel}_arrival_rayplot.png in the export/seismic folder")
    plt.show()

  # def arrival_timeplot(self):
  #   self.arrival.plot_times().savefig(f'{self.eventid}-{self.network}-{self.station}-{self.channel}_arrival_timeplot.png',dpi=300)
  #   print(f"Time plot saved as {self.eventid}-{self.network}-{self.station}-{self.channel}_arrival_timeplot.png")
  #   plt.show()