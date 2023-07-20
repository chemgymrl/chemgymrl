# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import sys
sys.path.append("../")


######################### Adding Some extra documentation #################
from chemistrylab.util.ActionDoc import append_doc
from chemistrylab.benches.general_bench import GenBench
import importlib
import gymnasium as gym
doc = set()
for env,reg in gym.registry.items():
    try:
        mod_name, attr_name = reg.entry_point.split(":")
        mod = importlib.import_module(mod_name)
        bench = getattr(mod, attr_name)
    except:
        bench = type(None)
    if issubclass(bench,GenBench) and not bench in doc:
        append_doc(bench)
        doc.add(bench)

############################################################################

#sys.path.append("C:\\Users\\sprag\\Documents\\Python Scripts\\ResearchF2022\\CHEMGYM\\chemgymrl")

project = 'ChemGymRL'
copyright = '2023, Chris Beeler, Sriram Ganapathi Subramanian, Kyle Sprague, Nouha Chatti, ColinBellinger, Mitchell Shahen, Nicholas Paquin, Mark Baula, Amanuel Dawit, Zihan Yang,Xinkai Li, Mark Crowley, Isaac Tamblyn'
author = 'Chris Beeler, Sriram Ganapathi Subramanian, Kyle Sprague, Nouha Chatti, ColinBellinger, Mitchell Shahen, Nicholas Paquin, Mark Baula, Amanuel Dawit, Zihan Yang,Xinkai Li, Mark Crowley, Isaac Tamblyn'
release = '1.5.8'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    "sphinx.ext.napoleon",
    "myst_parser",]


source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', "_inventory/python.inv"),
    'pandas': ('https://pandas.pydata.org/docs/', '_inventory/pandas.inv'),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


def setup(app):
   app.add_css_file('css/custom.css')