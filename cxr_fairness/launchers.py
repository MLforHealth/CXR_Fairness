import subprocess
import sys
import unicodedata
import getpass
import time
from pathlib import Path
import os

def local_launcher(commands):
    for cmd in commands:
        subprocess.call(cmd, shell=True)
                
def slurm_launcher(commands, max_slurm_jobs, output_dirs):
    for output_dir, cmd in zip(output_dirs, commands):
        block_until_running(max_slurm_jobs, getpass.getuser())
        # out_str = subprocess.call(cmd, shell=True)
        out_str = subprocess.run(cmd, shell = True, stdout=subprocess.PIPE).stdout.decode(sys.stdout.encoding)
        print(out_str.strip())
        if output_dir:
            try:
                job_id = int(out_str.split(' ')[-1])
            except (IndexError, ValueError, AttributeError):
                print("Error in Slurm submission, exiting." )
                sys.exit(0)
            
            (Path(output_dir)/'job_id').write_text(str(job_id))


def get_slurm_jobs(user):
    # returns a list of jobs IDs for (queued and waiting, running)
    out = subprocess.run(['squeue -u ' + user], shell = True, stdout = subprocess.PIPE).stdout.decode(sys.stdout.encoding)
    a = list(filter(lambda x: len(x) > 0, map(lambda x: x.split(), out.split('\n'))))
    queued, running = [], []
    for i in a:
        if i[0].isnumeric():
            if i[4].strip() == 'PD':
                queued.append(int(i[0]))
            else:
                running.append(int(i[0]))
    return (queued, running)

def block_until_running(n, user):
    while True:
        queued, running = get_slurm_jobs(user)
        if len(queued) + len(running) < n:
            time.sleep(0.2)
            return True
        else:
            time.sleep(10)        
        
REGISTRY = {
    'local': local_launcher,
    'slurm': slurm_launcher
}