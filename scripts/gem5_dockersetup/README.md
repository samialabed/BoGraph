## Debugging container while running
You can ssh into container to see what is happening using this command
```bash
docker exec -it gem5aladdin /bin/bash
```

## Directory structure
The most interesting directory is going to be `workspace/gem5`.
This directory is where BoGraph places intermediate build artifactes and results of experiments.
Each experiment is stored in `/gem5/<name of exp>/<num of params>/<num of iter>/<name of optimizer>/<date>/<iter>/`

This might take too much space after a while, if you would like to clean the intermediate date (which should be backed up to host machine anyway) run this command:
* For Aladdin experiments
```bash
docker run --rm --mount source=gem5-aladdin-workspace,target=/workspace xyzsam/gem5-aladdin "rm -r gem5/"
```
* For SMAUG experiments:
```bash
docker run --rm --mount source=gem5-aladdin-workspace,target=/workspace xyzsam/smaug:latest "rm -r gem5/"
```
