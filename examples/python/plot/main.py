import matplotlib.pyplot as plt
import numpy as np
import wave
import sys

if len(sys.argv) < 2:
    print("Plots a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
    sys.exit(-1)


spf = wave.open(sys.argv[1], "r")

# Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.frombuffer(signal, dtype=np.int16)
fs = spf.getframerate()

# If Stereo
if spf.getnchannels() == 2:
    print("Just mono files")
    sys.exit(0)


Time = np.linspace(0, len(signal) / fs, num=len(signal))

plt.figure(1)
plt.title("Signal Wave...")
plt.plot(Time, signal)
plt.show()
