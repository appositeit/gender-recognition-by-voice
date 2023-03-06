import librosa
import pumpp
import numpy as np
from . import params as p

def get_pump(
        sr=p.SAMPLE_RATE,
        n_fft_secs=p.STFT_WINDOW_LENGTH_SECONDS,
        hop_length_secs=p.STFT_HOP_LENGTH_SECONDS,
        n_mels=p.NUM_MEL_BINS,
        fmax=p.MEL_MAX_HZ):

    mel = pumpp.feature.Mel(
        name='mel', sr=sr,
        n_mels=n_mels,
        n_fft=int(n_fft_secs * sr),
        hop_length=int(hop_length_secs * sr),
        fmax=fmax, log=True, conv='tf')

    return pumpp.Pump(mel)

def get_sampler(pump, n_frames=None, hop_duration=None):
    op = pump['mel']
    return pumpp.SequentialSampler(
        n_frames or p.NUM_FRAMES, *pump.ops,
        stride=op.n_frames(hop_duration or p.EXAMPLE_HOP_SECONDS))

def get_features(filename=None, y=None, sr=None, pump=None, sampler=None):
    pump = pump or get_pump()
    sampler = sampler or get_sampler(pump)
    return np.concatenate([
        x[p.PUMP_INPUT] for x in sampler(pump(filename, y=y, sr=sr))])

def get_timesteps(Z, pump=None, sampler=None):
    pump = pump or get_pump()
    sampler = sampler or get_sampler(pump)
    op = pump['mel']
    return librosa.core.frames_to_time(
        np.arange(len(Z)), op.sr, op.hop_length * sampler.stride)
