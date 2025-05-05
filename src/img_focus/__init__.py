# image_focus/__init__.py

__version__ = "0.1.0" # Doit correspondre à pyproject.toml

# Rend la classe Salency directement importable depuis le package
from .detector import Salency


__all__ = ['Salency', '__version__']