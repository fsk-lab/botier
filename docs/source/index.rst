.. _topics-index:

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
       :target: https://github.com/fsk-lab/botier/blob/main/LICENSE
       :alt: License
.. image:: https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white
       :target: https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white
       :alt: Supported versions
.. image:: https://readthedocs.org/projects/botier/badge/?version=latest
       :target: http://botier.readthedocs.io/?badge=latest
       :alt: Docs status
.. image:: https://github.com/fsk-lab/botier/workflows/Test/badge.svg
       :target: https://github.com/fsk-lab/botier/actions
       :alt: GitHub CI Action status
.. image:: https://img.shields.io/pypi/v/botier.svg
       :target: https://pypi.python.org/pypi/botier
       :alt: PyPI Package latest release
|
===============================
Welcome to BoTier Documentation
===============================

Next to the "primary" optimization objectives, optimization problems often contain a series of subordinate objectives, 
which can be expressed as preferences over either the outputs of an experiment, or the experiment inputs 
(e.g. to minimize the experimental cost). BoTier provides a flexible framework to express hierarchical user preferences 
over both experiment inputs and outputs.

BoTier is a lightweight plug-in for BoTorch, and can be readily integrated with the BoTorch ecosystem for Bayesian Optimization.

Getting started
===============

.. toctree::
   :caption: Getting started
   :hidden:

   intro/overview
   intro/installation

:doc:`intro/overview`
    What is BoTier and how can it help you?

:doc:`intro/installation`
    A step-by-step guide to installing BoTier and setting it up for your data processing tasks.


Usage
=====

.. toctree::
   :caption: Usage
   :hidden:

   usage/tutorial
   api_reference/modules

:doc:`usage/tutorial`
    Learn how to create and run your first BoTier optimization campaign.

:doc:`api_reference/modules`
    Explore the documentation for each submodule and its functionalities.



.. toctree::
   :caption: Citation
   :hidden:

   citation/cite