#!/bin/bash
# shellcheck disable=SC2164
BOGRAPH_DIR=$(dirname "$(pwd)")

sudo apt install openjdk-17-jdk openjdk-17-jre -y

cd "$HOME"

git clone --depth 1 https://github.com/cmu-db/benchbase.git
cd benchbase
./mvnw clean package -P postgres

cd target
tar xvzf benchbase-postgres.tgz

mv benchbase-postgres "$BOGRAPH_DIR"/autorocks/envs/postgres/benchmarks/benchbase/compiled

tmux