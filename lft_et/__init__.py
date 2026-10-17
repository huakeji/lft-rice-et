"""Seven-day reference and rice evapotranspiration forecasts from public weather forecast messages."""
from .features import Message, Station, build_sequence, rain_level
from .predict import Forecaster, load_stations, sequences_from_archive

__version__ = "1.0.0"
