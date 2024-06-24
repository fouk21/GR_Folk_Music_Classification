# GR_Folk_Music_Classification

Harmonies of Heritage: Classifying Traditional Greek Folk Songs by Region using Deep Learning Methods

<!-- https://badgen.net/badge/:subject/:status/:color?icon=github -->
![python version](https://badgen.net/badge/python/3.11/blue)
![pre-commit](https://badgen.net/badge/pre-commit/3.6.0/green)

## Report

[Find the report here.](Report.pdf)

## Presentation

[Find the presentation here.]('Harmonies of Heritage.pptx')

## Development

Install `pre-commit` to keep git commits in line:

    pip3 install pre-commit

## Visdom

Before running the CNN train/tests, start the `visdom` server:

    python -m visdom.server

## FastApi Application setup

1. Install `fastapi` and `toml`

        pip3 install fastapi
        pip3 install toml

2. Start the server

        fastapi dev src/index.py

3. Browse to the Swagger documentation:
    [Docs](http://127.0.0.1:8000/docs "Swagger Docs")
