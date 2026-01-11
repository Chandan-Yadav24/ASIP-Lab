# demos/dsp_utils.py
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import streamlit as st
import io

# --- Signal Class Hierarchy --------------------------------------------------

class Signal:
    """Base class for all signals."""
    def __add__(self, other):
        if other == 0:
            return self
        return SumSignal(self, other)

    def make_wave(self, duration=1, start=0, framerate=11025):
        """Samples the signal."""
        n = int(duration * framerate)
        ts = start + np.arange(n) / framerate
        ys = self.evaluate(ts)
        return Wave(ys, ts, framerate)

    def evaluate(self, ts):
        """Evaluates the signal at the given times."""
        raise NotImplementedError

class Sinusoid(Signal):
    """Represents a sinusoidal signal."""
    def __init__(self, freq=440, amp=1.0, offset=0, func=np.sin):
        self.freq = freq
        self.amp = amp
        self.offset = offset
        self.func = func

    def evaluate(self, ts):
        phases = 2 * np.pi * self.freq * ts + self.offset
        ys = self.amp * self.func(phases)
        return ys

class SumSignal(Signal):
    """Represents the sum of two signals."""
    def __init__(self, *signals):
        self.signals = signals

    def evaluate(self, ts):
        ys = np.sum([sig.evaluate(ts) for sig in self.signals], axis=0)
        return ys
    
    def __add__(self, other):
        # Flatten sums for efficiency if adding another sum
        if isinstance(other, SumSignal):
            return SumSignal(*(self.signals + other.signals))
        return SumSignal(*(self.signals + (other,)))

class WhiteNoise(Signal):
    """Represents uncorrelated Gaussian noise."""
    def __init__(self, amp=1.0):
        self.amp = amp

    def evaluate(self, ts):
        ys = np.random.normal(0, self.amp, len(ts))
        return ys

class UncorrelatedUniformNoise(Signal):
    """Represents uncorrelated uniform noise (UU Noise)."""
    def __init__(self, amp=1.0):
        self.amp = amp

    def evaluate(self, ts):
        # Uniform logic: typically -amp to +amp
        ys = np.random.uniform(-self.amp, self.amp, len(ts))
        return ys

class UncorrelatedGaussianNoise(WhiteNoise):
    """Represents uncorrelated Gaussian noise (UG Noise). Alias for WhiteNoise."""
    pass

class BrownianNoise(Signal):
    """Represents Brownian noise (integrated white noise)."""
    def __init__(self, amp=1.0):
        self.amp = amp

    def evaluate(self, ts):
        # Brownian noise is cumulative sum of white noise
        white = np.random.normal(0, self.amp, len(ts))
        ys = np.cumsum(white)
        # Normalize to keep amplitude reasonable
        ys = ys / np.std(ys) * self.amp if np.std(ys) > 0 else ys
        return ys

class PinkNoise(Signal):
    """Represents pink noise (1/f noise) - approximation."""
    def __init__(self, amp=1.0, beta=1.0):
        self.amp = amp
        self.beta = beta  # spectral slope

    def evaluate(self, ts):
        # Generate white noise and filter in frequency domain
        N = len(ts)
        white = np.random.normal(0, 1, N)
        
        # FFT
        fft_white = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(N)
        
        # Apply 1/f^beta filter (avoid division by zero)
        freqs[0] = 1  # DC component
        fft_pink = fft_white / (freqs ** (self.beta / 2))
        
        # IFFT back
        ys = np.fft.irfft(fft_pink, n=N)
        
        # Normalize
        ys = ys / np.std(ys) * self.amp if np.std(ys) > 0 else ys
        return ys


# --- DSP Utilities (Spectrum, Wave) ------------------------------------------

class IntegratedSpectrum:
    """Represents an integrated spectrum (cumulative power)."""
    def __init__(self, cs, fs):
        self.cs = np.array(cs)  # cumulative sum (normalized 0 to 1)
        self.fs = np.array(fs)  # frequencies

    def plot(self):
        """Returns a matplotlib figure of the integrated spectrum."""
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.fs, self.cs, color='tab:orange', linewidth=2)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Cumulative Power (Normalized)")
        ax.set_title("Integrated Spectrum")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        # Add reference line for white noise
        ax.plot([self.fs[0], self.fs[-1]], [0, 1], 'k--', alpha=0.3, label='White Noise (Ideal)')
        ax.legend()
        
        fig.tight_layout()
        return fig

class Spectrum:
    """Mock thinkdsp.Spectrum."""
    def __init__(self, hs, fs, framerate):
        self.hs = np.array(hs)  # complex amplitudes
        self.fs = np.array(fs)  # frequencies
        self.framerate = framerate

    def plot(self, show_phase=False, high=None):
        """Returns a matplotlib figure of the spectrum."""
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Magnitude spectrum
        amps = np.abs(self.hs)
        
        # Limit frequency range if requested
        if high:
            idx = self.fs <= high
            fs_plot = self.fs[idx]
            amps_plot = amps[idx]
        else:
            fs_plot = self.fs
            amps_plot = amps
            
        ax.plot(fs_plot, amps_plot, color="tab:purple")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Magnitude Spectrum")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def low_pass(self, cutoff):
        """Attenuate frequencies above cutoff."""
        self.hs[np.abs(self.fs) > cutoff] = 0

    def high_pass(self, cutoff):
        """Attenuate frequencies below cutoff."""
        self.hs[np.abs(self.fs) < cutoff] = 0

    def make_wave(self):
        """Inverse FFT to get back a Wave."""
        ys = np.fft.irfft(self.hs)
        ts = np.arange(len(ys)) / self.framerate
        return Wave(ys, ts, self.framerate)

    def make_integrated_spectrum(self):
        """Returns an IntegratedSpectrum (cumulative power)."""
        # Compute power (magnitude squared)
        power = np.abs(self.hs) ** 2
        
        # Cumulative sum
        cs = np.cumsum(power)
        
        # Normalize to 0-1
        if cs[-1] > 0:
            cs = cs / cs[-1]
        
        return IntegratedSpectrum(cs, self.fs)


class Wave:
    """Mock thinkdsp.Wave."""
    def __init__(self, ys, ts, framerate):
        self.ys = np.array(ys)
        self.ts = np.array(ts)
        self.framerate = framerate

    @property
    def start(self):
        return self.ts[0] if len(self.ts) > 0 else 0

    @property
    def end(self):
        return self.ts[-1] if len(self.ts) > 0 else 0

    @property
    def duration(self):
        return len(self.ys) / self.framerate if self.framerate else 0

    def scale(self, factor):
        """Multiplies yields by factor."""
        self.ys = self.ys * factor

    def shift(self, offset):
        """Shifts the wave in time by adding offset to ts."""
        self.ts = self.ts + offset

    def write(self, filename='output.wav'):
        """Writes the wave to a WAV file (simulated via bytes return usually)."""
        # Scale to 16-bit integer
        max_val = np.max(np.abs(self.ys))
        if max_val == 0:
            scaled = self.ys.astype(np.int16)
        else:
            scaled = np.int16(self.ys / max_val * 32767)
        wav.write(filename, self.framerate, scaled)

    def get_audio_bytes(self):
        """Returns the wave data as bytes (WAV format)."""
        buffer = io.BytesIO()
        # Scale to 16-bit integer for compatibility
        max_val = np.max(np.abs(self.ys))
        if max_val == 0:
            scaled = self.ys.astype(np.int16)
        else:
            # Normalize to avoid clipping or being too quiet
            scaled = np.int16(self.ys / max_val * 32767)
        
        wav.write(buffer, self.framerate, scaled)
        return buffer

    def plot(self):
        """Returns a matplotlib figure of the waveform."""
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(self.ts, self.ys, color="tab:blue", linewidth=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Waveform")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def make_spectrum(self):
        """Returns a Spectrum object."""
        hs = np.fft.rfft(self.ys)
        fs = np.fft.rfftfreq(len(self.ys), d=1.0/self.framerate)
        return Spectrum(hs, fs, self.framerate)

def read_wave(file_like):
    """Reads a WAV file and returns a Wave object."""
    try:
        framerate, data = wav.read(file_like)
        
        # Handle stereo (take one channel)
        if len(data.shape) > 1:
            data = data[:, 0]
            
        # Normalize to -1..1
        if data.dtype == np.int16:
            ys = data / 32768.0
        elif data.dtype == np.int32:
             ys = data / 2147483648.0
        elif data.dtype == np.uint8:
             ys = (data - 128) / 128.0
        else:
             ys = data # assume float

        duration = len(ys) / framerate
        ts = np.linspace(0, duration, len(ys), endpoint=False)
        
        return Wave(ys, ts, framerate)
    except Exception as e:
        st.error(f"Error reading WAV file: {e}")
        return None
