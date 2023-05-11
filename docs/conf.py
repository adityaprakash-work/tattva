"""Sphinx configuration."""
project = "Tattva"
author = "Aditya Prakash"
copyright = "2023, Aditya Prakash"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
