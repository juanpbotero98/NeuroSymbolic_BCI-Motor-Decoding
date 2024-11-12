# Neuro Symbolic for BCI Motor Decoding
## Datasets Information 
* Original dataset used in Nengo Summer School:
  [Neural population dynamics during reaching (Churchland et al,2018)](https://dandiarchive.org/dandiset/000070?search=000070&pos=1)
* IEEE BIOCAS 2024 Grand Challenge on Neural Decoding for Motor Control of Non-human Primates:
  [Nonhuman Primate Reaching with Multichannel Sensorimotor Cortex Electrophysiology(O'Doherty et al 2018)](https://zenodo.org/records/583331)- scroll to the bottom of the page to download the dataset, 24GB.
  or use [Neurobench](https://github.com/NeuroBench/neurobench) repository to download the dataset using the following command:
  ```
  from neurobench.datasets import PrimateReaching

  pr_dataset = PrimateReaching(file_path=file_path,
                             filename=filename,
                             num_steps=1,
                             train_ratio=0.5,
                             bin_width=0.004,
                             biological_delay=0,
                             download=True)
  ```

## NeuroBench Information: 
* Neurobench expects to recieve an snnTorch model, for such we will use the [NIR repo](https://github.com/neuromorphs/NIR) (Neuromorphic Intermediate Representation) which allows us to convert a Nengo model to a snnTorch model.

## Running Jobs on CHPC: 
### Load module:
```
  module load miniforge3/latest
```

### Utilizing 'notchpeak-gpu' in interactive mode:
* Check for free GPUs in the partition: 
  ```
    freegpus -p notchpeak-gpu
  ```

* Request GPU resources interactively:
  ```
    salloc -n 1 -N 1 -t 3:00:00 -p notchpeak-gpu -A notchpeak-gpu --gres=gpu:3090:1 --mem=32G
  ```
  - n 1: Requests 1 task (CPU)
  - N 1: Requests 1 node for the job
  - t 3:00:00: Sets the time limit to 3hr.
  - gres=gpu:t4:1: Requests 1 NVIDIA T4 GPU

* Run a batch job: 
  ```
    sbatch <path to .slurm file>
  ```
  Useful commands: 
    - scontrol show job <job ID>
    - squeue -u u1437983


### Further CHCP Resources: 
  * Running Jobs on GPUs via Slurm: [CHCP link](https://www.chpc.utah.edu/documentation/software/slurm-gpus.php)
  * Scheduling Jobs at the CHPC with Slurm: [CHCP link](https://www.chpc.utah.edu/documentation/software/slurm.php#aliases)
  * GPUs and Accelerators at the CHPC: [CHCP link](https://www.chpc.utah.edu/documentation/guides/gpus-accelerators.php)
  * Youtube channel: [Youtube](https://www.youtube.com/@centerforhighperformanceco2744)




  

