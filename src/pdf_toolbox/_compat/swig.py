from __future__ import annotations

import warnings

# PyMuPDF relies on SWIG bindings that currently emit `SwigPy* has no __module__ attribute`
# warnings on Python 3.11+ when built with SWIG 4.3.1 (see
# https://github.com/pymupdf/PyMuPDF/issues/3931 and https://github.com/swig/swig/issues/2881).
# macOS wheels cannot yet consume SWIG 4.4.0 due to https://github.com/swig/swig/issues/3279,
# so we silence the warning at import time to keep user logs clean until upstream releases catch up.
warnings.filterwarnings(
    "ignore",
    message=r".*Swig.* has no __module__ attribute",
    category=DeprecationWarning,
)
