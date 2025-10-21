"""Defines the plugin hooks for VITools using the `pluggy` library.

This module establishes the hook specifications that allow other packages to
extend VITools by registering new phantom types. The primary mechanism is the
`PhantomSpecs` class, which defines the `register_phantom_types` hook.

External plugins can implement this hook to make their custom `Phantom`
subclasses discoverable by the main application.

Attributes:
    PROJECT_NAME (str): The unique name for the pluggy plugin system.
    hookspec (pluggy.HookspecMarker): A decorator to mark hook specifications.
    hookimpl (pluggy.HookimplMarker): A decorator to mark hook implementations.
"""

import pluggy
from typing import Dict, Type
from .phantom import Phantom

PROJECT_NAME = "VITools"
hookspec = pluggy.HookspecMarker(PROJECT_NAME)
hookimpl = pluggy.HookimplMarker(PROJECT_NAME)


class PhantomSpecs:
    """A collection of hook specifications for phantom-related plugins.

    This class defines the contracts that plugins must adhere to when they
    want to register new phantom types with VITools.
    """

    @hookspec
    def register_phantom_types(self) -> Dict[str, Type[Phantom]]:
        """A hook for plugins to register their custom Phantom subclasses.

        Plugins should implement this hook to make their phantom types available
        to the `Study` class and other parts of the application. The hook
        implementation must return a dictionary where keys are unique string
        identifiers for the phantoms and values are the corresponding `Phantom`
        subclasses (not instances).

        Returns:
            Dict[str, Type[Phantom]]: A dictionary mapping phantom names to
            `Phantom` subclasses. An empty dictionary should be returned if a
            plugin has no types to register.
        """
        return {}
