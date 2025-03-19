import numpy as np
import librosa

"""
Get the onset signal strength using spectral flux
"""
def get_onset_strength_signal(snd, hop, fftsize):  
    
    # round up to next power of 2
    len1 = int(2 ** np.ceil(np.log2(fftsize)))
    len2 = int(len1/2)
    
    # centre first frame at t=0
    snd = np.concatenate([np.zeros(len2), snd, np.zeros(len2)])
    
    frameCount = int(np.floor((len(snd) - len1) / hop + 1))
    
    prevM = np.zeros(len1)
    odf_sf = np.zeros(frameCount)   
    
    for i in range(frameCount):
        start = i * hop
        currentFrame = np.fft.fft(np.multiply(snd[start: start+len1],np.hamming(len1)))
        mag = np.abs(currentFrame)
        sf = np.mean(np.multiply(np.greater(mag, prevM), np.subtract(mag, prevM)))
        odf_sf[i] = sf
        prevM = mag
        
    return odf_sf

"""
Estimate beat period using autocorrelation
"""
def get_beat_period(OSS, frame_rate):
        
    acf = np.correlate(OSS, OSS, mode='full')
    
    # only use positive lags
    acf = acf[len(acf)//2:]
    # set our min and max lags using the BPM
    min_lag = int(60 * frame_rate / 210)
    max_lag = int(60 * frame_rate / 60)

    # set the tempo range of our acf
    acf_range = acf[min_lag:max_lag + 1]  

    # beat period is our max lag
    beat_period = np.argmax(acf_range) + min_lag

    #tempo = 60 * frame_rate / beat_period
        
    return beat_period


"""
Apply a length-k median filter to a 1D array x.
Boundaries are extended by repeating endpoints.
"""
def medfilt(x, k):
   # assert k % 2 == 1, "Median filter length must be odd."
   if x.ndim > 1: # Input must be one-dimensional."
      y = []  #np.empty((0,100), float)
      xt = np.transpose(x)
      for i in xt:
         y.append(medfilt(i, k))
      return np.transpose(y)
   k2 = (k - 1) // 2
   y = np.zeros((len(x), k), dtype=x.dtype)
   y[:,k2] = x
   for i in range(k2):
      j = k2 - i
      y[j:,i] = x[:-j]
      y[:j,i] = x[0]
      y[:-j,-(i+1)] = x[j:]
      y[-j:,-(i+1)] = x[-1]
   return np.median(y, axis=1)

"""
Get the beat positions from the onset strength
signal using pulse train shifting

Inspired by methods in Alonso, Miguel & David, 
Bertrand & Richard, Ga¨el. (2004). Tempo and
Beat Estimation of Musical Signals
"""
def get_beat_positions(OSS, beat_period):   
    
    # align beats using phase Alignment
    sums = []

    # arrange beats spaced by beat_period
    # ensure within bounds
    # sum the odf values at the points of the beats
    for offset in range(beat_period):
        aligned_beats = np.arange(offset, len(OSS), beat_period)
        aligned_beats = aligned_beats[aligned_beats < len(OSS)]
        sums.append(np.sum(OSS[aligned_beats]))

    # use highest sum as our pulse train offset
    offset = np.argmax(sums)  # Optimal phase offset
    beat_frames = np.arange(offset, len(OSS), beat_period)
    beat_frames = beat_frames[beat_frames < len(OSS)]
    return beat_frames

"""
Snap the found beats to the nearsest beats within
the specified window size
"""
def snap_beats(OSS, beat_frames, snap_window_size):
    snapped_beats = []
    for beat in beat_frames:
        search_start = max(0, beat - snap_window_size)
        search_end = min(len(OSS), beat + snap_window_size)
    
        search_window = OSS[search_start:search_end+1]
    
        # find the index of the max value within the search range
        best_peak_index = np.argmax(search_window) + search_start
        snapped_beats.append(best_peak_index)
        
    return np.array(snapped_beats)

"""
returns a list of beats and down beats 
though down beat detection was not implemented
so down beats is in a empty array
"""
def beatTracker(audio_file):

    # load audio
    audio, sample_rate = librosa.load(audio_file, mono=True, sr=44100)

    # set parameters
    hop_size = 250
    frame_size = 1256
    med_filt_size = 3

    # frame rate, OSS frame length
    frame_rate = sample_rate / hop_size 

    # get onset strength signal
    OSS = get_onset_strength_signal(audio, hop_size, frame_size)

    # apply median filter
    OSS = medfilt(OSS,med_filt_size)

    # estimate tempo/beatperiod using auto correlation
    beat_period = get_beat_period(OSS, frame_rate)
    beat_frames = get_beat_positions(OSS, beat_period)

    snap_window_size = (int)(beat_period/8)
    snapped_beat_frames = snap_beats(OSS, beat_frames, snap_window_size)
    
    # Convert refined beat indices to time
    beats = (snapped_beat_frames* hop_size) / sample_rate
    downbeats = [] # i didnt do down beats
    return beats, downbeats