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

