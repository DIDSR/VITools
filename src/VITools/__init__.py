from .phantom import Phantom
from .scanner import Scanner, read_dicom, available_scanners
from .study import Study
from . import hooks

import pluggy


def get_available_phantoms():
    pm = pluggy.PluginManager(hooks.PROJECT_NAME)
    pm.add_hookspecs(hooks.PhantomSpecs)
    num_loaded = pm.load_setuptools_entrypoints(group=hooks.PROJECT_NAME)

    # --- Call the hook to get all registered phantom types ---
    # The hook returns a list of lists (one list per plugin implementation that returned something)
    list_of_results = pm.hook.register_phantom_types()
    # Flatten the list of lists and filter out None or empty lists from plugins
    discovered_phantom_classes = {}
    for result_list in list_of_results:
        if result_list:  # Check if the plugin returned a non-empty list
            discovered_phantom_classes.update(result_list)
    return discovered_phantom_classes


from .study import Study
