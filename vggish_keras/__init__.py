import librosa
import pumpp
import numpy as np

from .vggish import VGGish
from .pump import get_pump, get_sampler, get_features, get_timesteps
from . import params
p = params



def get_embeddings(filename=None, y=None, sr=None, **kw):
    model, pump, sampler = get_embedding_model(**kw)

    # compute model outputs
    X = get_features(filename, y, sr, pump=pump, sampler=sampler)
    Z = model.predict(X)
    return Z, get_timesteps(Z, pump, sampler)

def get_embedding_model(model=None, pump=None, sampler=None, hop_duration=None,
                        include_top=None, compress=None, weights=None,):
    # make sure we have model, pump, and sampler
    # get the sampler with the proper frame sizes
    pump = pump or get_pump()
    model = model or VGGish(
        pump, include_top=include_top, compress=compress, weights=weights)
    sampler = sampler or get_sampler(
        pump, n_frames=model.input_shape[1], hop_duration=hop_duration)
    return model, pump, sampler

def get_embedding_function(*a, **kw):
    import functools
    model, pump, sampler = get_embedding_model(*a, **kw)
    compute = functools.partial(get_embeddings, model=model, pump=pump, sampler=sampler)
    compute.model = model
    compute.pump = pump
    compute.sampler = sampler
    return compute
