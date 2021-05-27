import submitit

def add(a, b):
    return a + b

# executor is the submission interface (logs are dumped in the folder)
executor = submitit.AutoExecutor(folder="log_test")

# set timeout in min, and partition for running the job
num_gpus = 1
workers_per_gpu = 10

executor.update_parameters(
        timeout_min=60,
        gpus_per_node=num_gpus,
        slurm_additional_parameters={"account": "rrg-bengioy-ad"},
        tasks_per_node=num_gpus,
        cpus_per_task=workers_per_gpu,
        slurm_mem="16G",#16G
        slurm_array_parallelism=100,
    )

job = executor.submit(add, 5, 7)  # will compute add(5, 7)
print(job.job_id)  # ID of your job

output = job.result()  # waits for completion and returns output
assert output == 12  # 5 + 7 = 12...  your addition was computed in the cluster
print('Job Submitted')