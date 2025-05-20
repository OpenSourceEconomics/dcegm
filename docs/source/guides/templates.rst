.. _templates:


Using Templates
===============

We provide two templates to facilitate setting up models with `dc-egm`. The template models include a minimal working example ("simplemodel") for an easy start and a maximal working example ("fullmodel") which demonstrates a possible types of features that are currently supported by `dc-egm`.

In order to install a template, follow these steps:

1. Make sure you have `dc-egm` installed via pip.
2. In your terminal, navigate to the directory you want to create the template in.
3. Run the following command to install the template where `<template-project>` is the name of the folder you want to create and `<model>` is either "simplemodel" or "fullmodel":

   ```bash
   dcegm init <template-project> --style=<model>
   ```

The template contains example model specifications which are easy to adapt to your own model. We provide the code in a jupyter notebook and Python file which can be used based on personal preference.
