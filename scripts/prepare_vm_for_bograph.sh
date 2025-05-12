#!/bin/bash
# shellcheck disable=SC2164
BOGRAPH_DIR=$(dirname "$(pwd)")

cd "$HOME" || exit
# Install miniconda in silient mode
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
eval "$("$HOME"/miniconda/bin/conda shell.bash hook)"
conda init
source "$HOME"/.bashrc

# install docker
cd "$HOME" || exit
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo chmod 666 /var/run/docker.sock

# install python dependencies
sudo apt install swig python3-dev gcc -y      # dependency for SMAC tuner
sudo apt-get install graphviz graphviz-dev -y # dependency for pygraphviz
conda env create -f "$BOGRAPH_DIR"/frozen_env.yaml
conda activate bograph

cd "$BOGRAPH_DIR"
pip install -e .[dev]
