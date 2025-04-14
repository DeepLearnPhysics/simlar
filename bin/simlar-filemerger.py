#!/usr/bin/python

import h5py, os
import numpy as np
import glob

def main(output_file, *args):
    """
    Main function to merge HDF5 files containing variable-length compound arrays.

    Parameters:
    - args: list of str, paths to the HDF5 files to be merged.
    """
    if len(args) < 2:
        print("Please provide at least two HDF5 files to merge.")
        return

    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Please remove it or choose a different name.")
        return

    with h5py.File(output_file,'w') as fout:
        files = [h5py.File(file, 'r') for file in args]
        meta_map = {str(key):None for key in files[0].attrs.keys()}
        data_map = {}
        def list_datasets(name,obj):
            if isinstance(obj, h5py.Dataset):
                data_map[name] = []
        files[0].visititems(list_datasets)
        for f in files:
            print('\nProcessing file:', f.filename)
            for key in data_map:
                if key in f:
                    data_map[key].append(np.array(f[key]))
                    print(f"    Key {key} data length: {len(data_map[key][-1])}")
                else:
                    raise ValueError(f"Key {key} not found in {f.filename}")
            for key in meta_map:
                if key in f.attrs:
                    if meta_map[key] is None:
                        meta_map[key]=f.attrs[key]
                    elif not key == 'config':
                        if type(meta_map[key]) in [type(dict()), type(str())]:
                            valid = (meta_map[key] == f.attrs[key])
                        elif type(meta_map[key]) == np.ndarray:
                            valid = np.array_equal(meta_map[key], f.attrs[key])
                        else:
                            raise ValueError(f'Unsupported type for attribute {key}: {type(meta_map[key])}')

                        if not valid:
                            print('Reference....')
                            print(meta_map[key])
                                
                            print('\nSubject...')
                            print(f.attrs[key])
                            
                            raise ValueError(f"Attribute {key} contents mismatch for file: {f.filename}")                       
                else:
                    print(f"Attribute {key} not found in {f.filename}")
                    print(f"Removing from the output file")
                    del meta_map[key]
        num_events = 0
        print()
        for key in data_map:
            print('Storing data for key:', key)
            # Concatenate variable-length arrays
            merged_data = np.concatenate(data_map[key],axis=0)
            if num_events < 1:
                num_events = len(merged_data)
            else:
                if num_events != len(merged_data):
                    raise ValueError(f"Dataset {key} has inconsistent length",len(merged_data),"across files",num_events)
            # Create a dataset in the output file with the same dtype
            fout.create_dataset(key, data=merged_data, dtype=files[0][key].dtype)
        for key in meta_map:
            print('Storing attribute for key:', key)
            # Create an attribute in the output file
            fout.attrs[key] = meta_map[key]

        print('Stored', num_events, 'events in the output file:', output_file)
        for f in files:
            f.close()

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Merge HDF5 files containing variable-length compound arrays.')
    parser.add_argument('output_file', type=str, help='Output file name')
    parser.add_argument('input_files', nargs='+', help='Input HDF5 files to merge')

    args = parser.parse_args()

    flist = []
    for expr in args.input_files:
        local_flist = glob.glob(expr)
        if len(local_flist) < 1:
            print(f"Error: No files found matching {expr}")
            sys.exit(1)
        flist.extend(local_flist)

    main(args.output_file, *flist)
