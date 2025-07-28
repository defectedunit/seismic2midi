# seismic2midi
<img src="https://github.com/defectedunit/seismic2midi/blob/main/seismic2MIDI.png" alt="seismic2midi" width="80" height="80"> Open source Seismic Data Sonification package 

## Check out the slidedeck of seismic2midi:
<img src="https://www.freeiconspng.com/thumbs/video-icon/video-icon-1.png" alt="Watch the slide deck of seismic2midi" width="80" height="80"> 
<a href="https://www.canva.com/design/DAGucwk1Nr4/n9h0uzL5_eqaWdwDu2hl8Q/watch?utm_content=DAGucwk1Nr4&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=hc2aebcd539"><img alt="Watch the slide deck of seismic2midi" src="https://www.freeiconspng.com/thumbs/video-icon/video-icon-1.png style="width:42px;height:42px;"/></a>



<details open> 
  <summary><h2>Exports</h2></summary>

  All the files you can want for is here, including:

  - midi file (.mid)
  - audio file (.wav)
  - plots (event location, fft, spectrogram, phase arrivals, score)
  - metadata (.csv and .parquet)  
</details>



## sesimic2midi( ) - a package for seismic data to midi and audio conversion (also known as data sonification)
=======================================================

Simply by providing an eventid of a seismic event, sesimic2midi can spit out visual, midi and audio representations of the event.
Data are stored at ./exports/seismic and ./exports/music, with various variables to parse in.

.. Note::
    To find Earthquake events, you can use the IRIS FDSN client.
    See https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.client.Client.html#obspy.clients.fdsn.client.Client.get_events

    Or Here for laypeople query: https://earthquake.usgs.gov/earthquakes/browse/significant.php?year=2011

    For API, you will have to these install dependencies:
    # !pip install cartopy
    # !pip install obspy
    # import cartopy.crs as ccrs
    # import obspy
    # from obspy import UTCDateTime

    This code will return all the events with a magnitude of 7 or higher.
    # event_list = client.get_events(minmagnitude=7.0)
      event_list

    This code will return all the events with a magnitude of 7 or higher during 2024, and .plot() will visualize their locations on a map with robbinson projection.
    # event_list = client.get_events(minmagnitude=7.0, starttime=UTCDateTime("2024-01-01"),endtime=UTCDateTime("2025-01-01"))
    # event_list.plot()


.. Event and trace variables::

    client: default IRIS, for the list of available clients, see https://docs.obspy.org/packages/obspy.clients.fdsn.html
    network, station, location, channel: default: 'IU', 'ANMO', '00', 'BHZ' they are for specifying a specific trace's station's channel see https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.client.Client.html#obspy.clients.fdsn.client.Client.get_waveforms
    event_summary (bool): retruns a dataframe and stored as ./exports/seismic/event_summary.parquet

.. Seismic variables::

    model: default TauPyModel(model="iasp91"), see https://docs.obspy.org/packages/obspy.taup.html
    location_plot (bool): default False. Retruns a location plot and stored as ./exports/seismic/f'{filename}_location_plot.png'
    fft (bool): retruns a time series plot and stored as ./exports/seismic/f'{filename}_fft.png
    spectrogram (bool): retruns a spectrogram and stored as ./exports/seismic/f'{filename}_spectrogram.png'
    model: default TauPyModel(model="iasp91"), see https://docs.obspy.org/packages/obspy.taup.html

    Note on arrvial: Phase arrivals are calculated using the TauPy model, around 20-40 arrivals per event.
    arrival_rayplot (bool): default False. Retruns a rayplot and stored as ./exports/seismic/f'{filename}_arrival_rayplot.png'

.. Music variables::

    fundamental: defualt "C4", by specifying the f0 you can shift the note numbers of the event.
    tempo_bpm: default 120 Beats Per Minute. See https://en.wikipedia.org/wiki/Beats_per_minute
    sperical_coordinates: Retruns a dataframe and stored as ./exports/music/eventid_sperical_coordinates.parquet
    midifile (bool): default True. Returns a MIDI file and stored as ./exports/music/f'{filename}_music/.mid'
    score (bool): default True. Returns a score and stored as ./exports/music/f'{filename}_score/.png'
    audiofile (bool): default False. A preview is provided but you are engouraged to use the midi file in your DAW. Returns a audio file and stored as ./exports/music/f'{filename}_music/.wav'

.. filename::

    Default filename = f'{eventid}_{network}_{station}_{channel}'. Feel free to modify as you see fit.


:copyright:
    Jeremy Leung (info.sostenuto@gmail.com, IG, Github: @defectedunit)

    
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
