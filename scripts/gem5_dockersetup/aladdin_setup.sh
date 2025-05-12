#!/bin/bash
CONTAINER_NAME=gem5_build
IMAGE_NAME=xyzsam/gem5-aladdin:latest
docker pull $IMAGE_NAME
# Start build container in the background
docker run -dit --name=$CONTAINER_NAME --mount source=gem5-aladdin-workspace,target=/workspace $IMAGE_NAME
# copy the build scripts
docker cp ./docker_gem5_build_cmd.sh $CONTAINER_NAME
# execute the build command, might take a while
docker exec -it $CONTAINER_NAME sh -c "./docker_gem5_build_cmd.sh"

docker kill $CONTAINER_NAME
docker rm $CONTAINER_NAME