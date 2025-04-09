#!/usr/bin/env python3
import torch
import yaml
import fire
import os
import simlar
from simlar.utils import list_config, build_config
from simlar.simulate import DetectorSimulator

def main(config,
         device=None,
         input_file=None,
         output_file=None,
         num_entries=-1,
         num_skip=-1,
         ):

    print(f"\nUsing simlar version: {simlar.__version__}")

    if not os.path.isfile(input_file):
        raise ValueError(f"Input file '{input_file}' not found.")

    if os.path.isfile(output_file):
        raise ValueError(f"Output file '{output_file}' already exists.")

    try:
        num_entries = int(num_entries)
    except ValueError:
        raise ValueError(f"num_entries '{num_entries}' must be an integer.")
    try:
        num_skip = int(num_skip)
    except ValueError:
        raise ValueError(f"num_skip '{num_skip}' must be an integer.")

    if num_entries < -1:
        raise ValueError(f"num_entries '{num_entries}' must be -1 or greater.")
    if num_skip < -1:
        raise ValueError(f"num_skip '{num_skip}' must be -1 or greater.")

    # check the configuration structure
    print('Building configuration:',config)
    config = build_config(config)
    # update run flags
    config['SIMULATION']['IO']['input_file'] = input_file
    config['SIMULATION']['IO']['output_file'] = output_file
    config['SIMULATION']['IO']['num_entries'] = num_entries
    config['SIMULATION']['IO']['num_skip'] = num_skip

    print('\nRun configuration below...')
    print(yaml.dump(config, default_flow_style=False))

    print('-------------------------------------------------------')
    print('Instantiating detector simulator...')
    print('-------------------------------------------------------\n')

    simulator = DetectorSimulator(config)
    simulator.process()

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    fire.Fire(main)


