"""PyAudio Example: Play a wave file (callback version)"""

import wave
import time
import sys

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import wavio
import scipy
from scipy import signal
import numba
import soundfile as sf


def pcm_to_float(sig, dtype='float32'):
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
    float_to_pcm, dtype
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


def float_to_pcm(sig, dtype='int16'):
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
    pcm_to_float, dtype
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
    return float_to_pcm(sig, dtype='int16').tobytes()


def byte_to_float(byte):
    # byte -> int16(PCM_16) -> float32
    return pcm_to_float(np.frombuffer(byte, dtype=np.int16), dtype='float32')


class Processor:
    def __init__(self):
        pass

    def prepare(self, sample_rate, block_size, num_channels):
        raise NotImplementedError()

    def process(self, buffer):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


class Gain(Processor):
    def __init__(self, gain):
        self.gain = gain

    def prepare(self, sample_rate, block_size, num_channels):
        pass

    def process(self, buffer):
        return buffer * self.gain

    def reset(self):
        pass


class Plot(Processor):
    def __init__(self, filename):
        self.plot_buffer = np.zeros(128, dtype='float32')
        self.sample_rate = 44100.0
        self.filename = filename

    def prepare(self, sample_rate, block_size, num_channels):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.num_channels = num_channels

    def process(self, buffer):
        self.plot_buffer = np.append(self.plot_buffer, buffer)
        return buffer

    def reset(self):
        end_time = len(self.plot_buffer) / self.sample_rate
        time_axis = np.linspace(0, end_time, num=len(self.plot_buffer))

        plt.figure(1)
        plt.title(self.filename)
        plt.plot(time_axis, self.plot_buffer)
        plt.savefig(self.filename)


class CombFilter:
    def set_size(self, size):
        self.buffer_index = 0
        self.buffer_size = size
        self.clear()

    def clear(self):
        self.last = 0.0
        self.buffer = np.zeros(self.buffer_size, dtype='float32')

    def process_sample(self, input, damp, feedback):
        output = self.buffer[self.buffer_index]
        self.last = (output * (1.0 - damp)) + (self.last * damp)

        temp = input + (self.last * feedback)
        self.buffer[self.buffer_index] = temp
        self.buffer_index = (self.buffer_index + 1) % self.buffer_size
        return output


class AllPassFilter:
    def set_size(self, size):
        self.buffer_index = 0
        self.buffer_size = size
        self.clear()

    def clear(self):
        self.buffer = np.zeros(self.buffer_size, dtype='float32')

    def process_sample(self, input):
        bufferedValue = self.buffer[self.buffer_index]
        temp = input + (bufferedValue * 0.5)
        self.buffer[self.buffer_index] = temp
        self.buffer_index = (self.buffer_index + 1) % self.buffer_size
        return bufferedValue - input


class FreeVerb(Processor):
    def __init__(self):
        self.room_size = 0.5     # Room size, where 1 is big, 0 is small.
        self.damp = 0.5     # Damp, where 0 is no damped, 1 is full damped.
        self.wet_level = 0.33    # Wet level, 0 to 1
        self.dry_level = 0.4     # Dry level, 0 to 1
        self.width = 1.0     # Reverb width, where 1 is very wide.
        self.freeze_mode = 0.0     # Freeze mode - values < 0.5 are "normal" mode, values > 0.5

    def prepare(self, sample_rate, block_size, num_channels):
        pass

    def process(self, buffer):
        return buffer * self.gain

    def reset(self):
        pass


@numba.njit()
def firstorder_tpt_process_sample(input, G, s, type):
    v = G * (input - s)
    y = v + s
    s = y + v

    out = y
    if type == 1:  # low pass
        out = y
    if type == 2:  # high pass
        out = input - y
    if type == 3:  # all pass
        out = 2 * y - input
    return (out, s)


@numba.njit()
def firstorder_tpt_process(buffer, G, s, type):
    for i in range(buffer.shape[0]):
        out, new_s = firstorder_tpt_process_sample(buffer[i], G, s, type)
        buffer[i] = out
        s = new_s
    return (buffer, s)


class FirstOrderTPTFilter(Processor):
    def __init__(self, freq, type):
        self.frequency = freq
        self.type = type

    def prepare(self, sample_rate, block_size, num_channels):
        self.sample_rate = sample_rate
        self.reset()
        self.update()

    def process(self, buffer):
        G = self.G
        type = self.type
        s = self.s1[0]
        buffer, s = firstorder_tpt_process(buffer, G, s, type)
        self.s1[0] = s
        return buffer

    def reset(self):
        self.s1 = np.zeros(1, dtype='float32')

    def update(self):
        freq = self.frequency
        sample_rate = self.sample_rate
        g = np.tan(np.pi * freq / sample_rate)
        self.G = g / (1 + g)


class AudioApplication(Processor):
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.wav = None

    def __del__(self):
        self.pa.terminate()

    def prepare_file(self, file):
        self.wav = wave.open(file, 'rb')

        self.format = self.pa.get_format_from_width(self.wav.getsampwidth())
        self.sample_rate = self.wav.getframerate()
        self.block_size = 1024
        self.num_channels = self.wav.getnchannels()
        self.prepare(self.sample_rate, self.block_size, self.num_channels)

        print(f'Fs: {self.sample_rate}, Ch: {self.num_channels}')

    def release_file(self):
        self.wav.close()
        self.reset()

    def play_file(self, file):
        self.prepare_file(file)

        def callback(in_data, frame_count, time_info, status):
            buffer = byte_to_float(self.wav.readframes(frame_count))
            buffer = self.process(buffer)
            return (float_to_byte(buffer), pyaudio.paContinue)

        stream = self.pa.open(format=self.format,
                              channels=self.num_channels,
                              rate=self.sample_rate,
                              output=True,
                              stream_callback=callback,
                              frames_per_buffer=self.block_size)

        stream.start_stream()

        while stream.is_active():
            time.sleep(0.1)

        stream.stop_stream()
        stream.close()
        self.release_file()

    def render_file(self, in_file, out_file):
        info = sf.info(in_file)

        self.format = info.format
        self.subtype = info.subtype
        self.sample_rate = info.samplerate
        self.block_size = 1024
        self.num_channels = info.channels
        self.prepare(self.sample_rate, self.block_size, self.num_channels)

        render_buffer = np.zeros(0, dtype='float32')
        for block in sf.blocks(in_file, blocksize=self.block_size, dtype='float32'):
            render_buffer = np.append(render_buffer, self.process(block))

        self.reset()
        sf.write(out_file, render_buffer, self.sample_rate, self.subtype)


class ExampleApp(AudioApplication):
    def __init__(self):
        super().__init__()
        self.filter = FirstOrderTPTFilter(220.0, 1)
        self.gain = Gain(0.95)

    def prepare(self, sample_rate, block_size, num_channels):
        self.filter.prepare(sample_rate, block_size, num_channels)
        self.gain.prepare(sample_rate, block_size, num_channels)

    def process(self, buffer):
        buffer = self.filter.process(buffer)
        buffer = self.gain.process(buffer)
        return buffer

    def reset(self):
        self.filter.reset()
        self.gain.reset()


def main():
    if len(sys.argv) < 2:
        print(f'Plays a wave file.\n\nUsage: {sys.argv[0]} filename.wav')
        sys.exit(1)

    app = ExampleApp()
    # app.play_file(sys.argv[1])
    app.render_file(sys.argv[1], "myfile.wav")


if __name__ == "__main__":
    main()
