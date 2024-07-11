#!/bin/bash
sudo apt update
sudo apt install -y libopenblas-dev libyaml-dev ffmpeg

# Install poetry
pipx install poetry --pip-args '--no-cache-dir --force-reinstall'

# Install project dependencies
poetry install
