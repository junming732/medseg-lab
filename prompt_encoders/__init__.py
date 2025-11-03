from .base_encoder import BasePromptEncoder
from .binary_encoder import BinaryPromptEncoder
from .gaussian_encoder import GaussianPromptEncoder
from .disk_encoder import DiskPromptEncoder
from .geodesic_encoder import GeodesicPromptEncoder
from .sam_encoder import SAMPromptEncoder

__all__ = [
    'BasePromptEncoder',
    'BinaryPromptEncoder',
    'GaussianPromptEncoder',
    'DiskPromptEncoder',
    'GeodesicPromptEncoder',
    'SAMPromptEncoder',
]

def get_encoder(encoder_type: str, **kwargs):
    encoders = {
        'binary': BinaryPromptEncoder,
        'gaussian': GaussianPromptEncoder,
        'disk': DiskPromptEncoder,
        'geodesic': GeodesicPromptEncoder,
        'sam': SAMPromptEncoder,
    }
    if encoder_type not in encoders:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    return encoders[encoder_type](**kwargs)