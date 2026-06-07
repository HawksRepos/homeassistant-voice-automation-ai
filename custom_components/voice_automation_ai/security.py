"""Shared security helpers: safe YAML loading and blocked-service scanning.

Centralised here so the conversation agent, the service handlers, and the file
manager all share one implementation. Security-critical logic must live in a
single place - a fix applied to a copy that another module doesn't use is worse
than no fix at all.
"""
from __future__ import annotations

from typing import Any

import yaml

from .const import BLOCKED_SERVICE_DOMAINS


class InputLoader(yaml.SafeLoader):
    """YAML loader that treats Home Assistant ``!input`` tags as plain strings.

    Blueprints use ``!input`` tags that the stock ``SafeLoader`` cannot
    construct. Loading a blueprint with ``yaml.safe_load`` therefore raises,
    and a caller that swallows that error would skip the security scan
    entirely. This loader keeps the document parseable so the scan can run.
    """


def _input_constructor(loader: yaml.Loader, node: yaml.Node) -> str:
    """Render an ``!input <name>`` tag back to a plain string."""
    return f"!input {loader.construct_scalar(node)}"


InputLoader.add_constructor("!input", _input_constructor)


def check_yaml_for_blocked_services(data: Any) -> str | None:
    """Recursively scan parsed YAML for references to blocked service domains.

    Walks every nested dict and list. On each dict it checks the ``service`` and
    ``action`` keys (Home Assistant accepts either) for a domain listed in
    :data:`BLOCKED_SERVICE_DOMAINS`. Because the walk descends into *all* values,
    blocked services nested inside ``choose``/``parallel``/``if``-``then``/
    ``repeat``/``sequence`` blocks are caught without enumerating those keys.

    Returns an error message if a blocked service is found, else ``None``.
    """
    if isinstance(data, dict):
        for key in ("service", "action"):
            svc = data.get(key)
            if isinstance(svc, str) and "." in svc:
                domain = svc.split(".")[0]
                if domain in BLOCKED_SERVICE_DOMAINS:
                    return (
                        f"Blocked: generated YAML references restricted "
                        f"service domain '{domain}'."
                    )
        for value in data.values():
            result = check_yaml_for_blocked_services(value)
            if result:
                return result
    elif isinstance(data, list):
        for item in data:
            result = check_yaml_for_blocked_services(item)
            if result:
                return result
    return None
