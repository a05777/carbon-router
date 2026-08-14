from .config import Settings
from .server import create_app


app = create_app(Settings.from_env())

