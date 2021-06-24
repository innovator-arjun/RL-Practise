import submitit
import wandb

def add(a):

    wandb.login()
# 1. Start a W&B run
    wandb.init(project='gpt6')

# 2. Save model inputs and hyperparameters
    config = wandb.config
    config.learning_rate = 0.01

# Model training code here ...

# 3. Log metrics over time to visualize performance
    for i in range (a):
        print(i)
        wandb.log({"loss": i})

    # executor is the submission interface (logs are dumped in the folder)
executor = submitit.AutoExecutor(folder="logs")

    # set timeout in min, and partition for running the job
num_gpus = 1

executor.update_parameters(
                            timeout_min=15,
                            gpus_per_node=num_gpus,
                            slurm_additional_parameters={"account": "def-bengioy"},
                             tasks_per_node=num_gpus,
                             slurm_mem="2G"#16G

                              )
job = executor.submit(add, 5)  # will compute add(5, 7)
print(job.job_id)  # ID of your job
print('Job Sucess')