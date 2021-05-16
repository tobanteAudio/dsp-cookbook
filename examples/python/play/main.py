"""PyAudio Example: Play a wave file (callback version)"""

import wave
import time
import sys

import numpy as np
import pyaudio


def pcm2float(sig, dtype='float32'):
    """Convert PCM signal to floating point with a range from -1 to 1.
    Use dtype='float32' for single precision.
    Parameters
    ----------
    sig : array_like
        Input array, must have integral type.
    dtype : data type, optional
        Desired (floating point) data type.
    Returns
    -------
    numpy.ndarray
        Normalized floating point data.
    See Also
    --------
    float2pcm, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max


def float2pcm(sig, dtype='int16'):
    """Convert floating point signal with a range from -1 to 1 to PCM.
    Any signal values outside the interval [-1.0, 1.0) are clipped.
    No dithering is used.
    Note that there are different possibilities for scaling floating
    point numbers to PCM numbers, this function implements just one of
    them.  For an overview of alternatives see
    http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html
    Parameters
    ----------
    sig : array_like
        Input array, must have floating point type.
    dtype : data type, optional
        Desired (integer) data type.
    Returns
    -------
    numpy.ndarray
        Integer data, scaled and clipped to the range of the given
        *dtype*.
    See Also
    --------
    pcm2float, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


def float_to_byte(sig):
    # float32 -> int16(PCM_16) -> byte
    return float2pcm(sig, dtype='int16').tobytes()


def byte_to_float(byte):
    # byte -> int16(PCM_16) -> float32
    return pcm2float(np.frombuffer(byte, dtype=np.int16), dtype='float32')


class Processor:
    def __init__(self):
        pass

    def prepare(self, sample_rate, block_size, num_channels):
        pass

    def process(self, buffer):
        return buffer


class Gain(Processor):
    def __init__(self, gain):
        self.gain = gain

    def process(self, buffer):
        buffer = buffer * self.gain
        return buffer


class AudioApplication:
    def __init__(self, dsp):
        self.pa = pyaudio.PyAudio()
        self.dsp = dsp

    def __del__(self):
        self.pa.terminate()

    def play_file(self, file):
        wav = wave.open(file, 'rb')

        format = self.pa.get_format_from_width(wav.getsampwidth())
        num_channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        print(f'Fs: {sample_rate}, Ch: {num_channels}, Format: {format}')

        self.dsp.prepare(sample_rate, 128, num_channels)

        def callback(in_data, frame_count, time_info, status):
            buffer = byte_to_float(wav.readframes(frame_count))
            buffer = self.dsp.process(buffer)
            return (float_to_byte(buffer), pyaudio.paContinue)

        stream = self.pa.open(format=format,
                              channels=num_channels,
                              rate=sample_rate,
                              output=True,
                              stream_callback=callback)

        stream.start_stream()

        while stream.is_active():
            time.sleep(0.1)

        stream.stop_stream()
        stream.close()
        wav.close()


if len(sys.argv) < 2:
    print(f'Plays a wave file.\n\nUsage: {sys.argv[0]} filename.wav')
    sys.exit(-1)

app = AudioApplication(Gain(0.5))
app.play_file(sys.argv[1])
