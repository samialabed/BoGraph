#!/bin/bash

# install docker
cd "$HOME" || exit
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo chmod 666 /var/run/docker.sock
docker pull postgres:alpine
# enable unsecure tls connection.
echo "Enabling unsecure TLS connection to docker, if you want a secure connection create a certification https://docs.docker.com/engine/security/protect-access/#connecting-to-the-secure-docker-port-using-curl"
sudo systemctl stop docker
#sudo dockerd -H unix:///var/run/docker.sock -H tcp://0.0.0.0:2376 --tls=false
echo '{"tls": false,"hosts": ["unix:///var/run/docker.sock", "tcp://0.0.0.0:2376"]}' | sudo tee -a "/etc/docker/daemon.json"
#https://stackoverflow.com/questions/44052054/unable-to-start-docker-after-configuring-hosts-in-daemon-json
# remove fd start
sudo cp /lib/systemd/system/docker.service /etc/systemd/system/
sudo sed -i 's/\ -H\ fd:\/\///g' /etc/systemd/system/docker.service
sudo systemctl daemon-reload
sudo service docker restart
sudo chmod 666 /var/run/docker.sock
echo "Finished preparing docker and now listening to remote connections on port 2376"