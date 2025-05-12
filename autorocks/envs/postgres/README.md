

Postgres plan:

* Run native vs docker?
  * Docker has overhead on write speed
  * Docker introduces latent variables we don't want
  * Docker is cleaner to install and ship around on multiple instances

* Benchmarking: Use BenchBase https://github.com/cmu-db/benchbase 

TODOs:
* [] Native vs Docker?
  - Will use docker for now
  - 
* [] Decide on parameters
* [] Docker compose be a python template and fill it  

## OtterTune paper setup
We use the same setup as OtterTune to compare against their results

OtterTuner Experiment setup:
* Used the OS's package manager to install the DBMS
* Changed it to allow remote access 
* Conducted the benchmark + OtterTune from separate instace

Benchmark used:
* OLTP-Bench (DEPRECATED)
* RanYCSB workload 

Instances setup:

OtterTune Paper setup: 

```
Each experiment consists of two instances. The first instance
is OtterTune’s controller that we integrated with the OLTP-Bench
framework. 
These clients are deployed on m4.large instances with
4 vCPUs and 16 GB RAM. The second instance is used for the tar-
get DBMS deployment. We used m3.xlarge instances with 4 vC-
PUs and 15 GB RAM. We deployed OtterTune’s tuning manager
and repository on a local server with 20 cores and 128 GB RAM.
```
Special versions:



Other consideration:
* Run PGTune and use it as baseline (https://pgtune.leopard.in.ua/) (https://github.com/le0pard/pgtune)


## How to update postgres configurations on docker


useful pages:
* https://dhamaniasad.github.io/awesome-postgres/#monitoring
