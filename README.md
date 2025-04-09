More verbose how to will follow...

## How to run G4 
- You would need a local build of `edep-sim` forked [here](https://github.com/drinkingkazu/edep-sim)
  - You can use the `develop.sif` neutrino image (as of 2025-04-08 pointing to `larcv2_ub2204-cuda121-torch251-larndsim.sif` )
- Necessary configuration files can be found under [generator](https://github.com/DeepLearnPhysics/simlar/tree/main/generator)

Example how to:
```
edep-sim -g geometry/BigLArCube.gdml -e 100 -o generator.h5 biglarbox.mac
```
Look though the contents of `biglarbox.mac`. For most users, the only part to change is the yaml file argument in the line below
```
/generator/kinematics/bomb/config mpvmpr_ccmu.yaml
```
which is a [DLPGenerator](https://github.com/DeepLearnPhysics/DLPGenerator) configuration yaml file.

### How to run a detector simulation

Example how to:
```
simlar-run.py run_develop --input_file generator.h5 --output_file detsim.h5
```


