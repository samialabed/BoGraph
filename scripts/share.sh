#!/bin/bash
# TODO: refactor this to run consolidation script then rsync the result

# sync local with remote
rsync -azP local_execution/ cam:experiment_res 
# grab remote to local
# rsync -azP cam:experiment_res/ local_execution
