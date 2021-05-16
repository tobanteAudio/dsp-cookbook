import wave
import sys


if len(sys.argv) < 2:
    print("Plots a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
    sys.exit(-1)


obj = wave.open(sys.argv[1], "r")

print("Number of channels", obj.getnchannels())
print("Sample width", obj.getsampwidth())
print("Frame rate.", obj.getframerate())
print("Number of frames", obj.getnframes())
print("parameters:", obj.getparams())
obj.close()
