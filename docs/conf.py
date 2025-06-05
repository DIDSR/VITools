# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'VITools'
copyright = '2025, Brandon J. Nelson'
author = 'Brandon J. Nelson'
release = '0.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import sys
from pathlib import Path

sys.path.insert(0, Path(__file__).parents[2])
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.duration',
              'sphinx.ext.doctest',
              'sphinx.ext.autosummary']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

html_theme_options = {
    'logo': 'VITools.png',
    'logo_name': 'true',
    'description': 'Tools for running virtual imaging trials',
    'font_family': 'Helvetica',
    'head_font_family': 'Helvetica',
    'font_size': '12pt',
    'fixed_sidebar': 'true',
    'page_width': '1200px',
    'sidebar_width': '250px'
}