
## Automatically
Run the `sudo sh install_postgres_docker.sh`

## Prepare docker daemon to accept remote executions manually

* Stop docker daemon on host machine: `sudo systemctl stop docker`
* Copy `daemon.json` to `/etc/docker/daemon.json` on Host
* Restart the daemon.

Be careful: make sure your host machine is behind a wall with very strict IP address policy

