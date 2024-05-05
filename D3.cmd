executable = scripts/run_main.sh
getenv = True
error = condor_logs/D3.error
log = condor_logs/D3.log
notification = always
transfer_executable = false
request_memory = 8*1024
request_GPUs = 1
+Research = True
Queue


