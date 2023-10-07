library(tuneR, warn.conflicts = F, quietly = T) # nice functions for reading and manipulating .wav files
library(Rssa)
# define path to audio file
fin = 'F:/Study/4-1/EE4000 Thesis/EE4000/klattSyn.wav'

# read in audio file
data = readWave(fin)

# extract signal
snd = data@left

# determine duration
dur = length(snd)/data@samp.rate
dur # seconds
## [1] 3.588

# determine sample rate
fs = data@samp.rate
fs # Hz
## [1] 2000
s=ssa(snd, 2000)
plot(s)
