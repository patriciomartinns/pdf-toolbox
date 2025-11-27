from __future__ import annotations

from . import schemas, services, tools
from ._compat import swig as _compat_swig
from .server import app, mcp

_ = _compat_swig

__all__ = ["app", "mcp", "schemas", "services", "tools"]
