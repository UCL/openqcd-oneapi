### Usage Notes

On the Cambridge CSD3 HPC cluster, please use the provided `Makefile.csd3` i.e. rename it to `Makefile`. This will load the necessary modules and set up the compilation environment for the different hardware targets available on this specific HPC cluster.

To run this on CSD3, please use the command:
```
make -f Makefile.csd3 <target_name>
```
where target_name is either:
- intel_cpu
- nvidia_gpu
