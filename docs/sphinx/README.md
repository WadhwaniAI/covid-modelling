# Sphinx Documentation

This folder contains the `.rst` files and `Makefile` to create sphinx documentation from the docstrings in the `.py` files of this repo.

To create the sphinx documentation, run `make html` or `make latexpdf` to create the documentation in HTML or PDF output format.

The sphinx documentation would be created in `./_build/html` or `./_build/latex` depending on your input to `make`. And don't worry, the created documentation will be ignored by git. 