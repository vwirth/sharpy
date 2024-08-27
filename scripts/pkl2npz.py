from __future__ import print_function
from lib2to3.pytree import convert
import numpy as np
import pickle
import sys
import os
from os import path


def convert_to_npz(input_dir):

    if (not os.path.isdir(input_dir)):
        print("Invalid input directory: {}".format(input_dir))
        exit(-1)

    output_dir = os.path.join(input_dir, "npz")

    counter = 0
    for pkl_path_rel in os.listdir(input_dir):
        pkl_path = os.path.join(input_dir, pkl_path_rel)
        if (os.path.isdir(pkl_path)):
            convert_to_npz(pkl_path)
        if (not pkl_path.endswith(".pkl")):
            continue

        if (counter == 0):
            os.makedirs(output_dir, exist_ok=True)

        print("converting: {}....".format(pkl_path))
        with open(pkl_path, 'rb') as pkl_file:
            try:
                pkl_data = pickle.load(pkl_file, encoding='latin1')
            except:
                pkl_data = pickle.load(pkl_file)

            output_data = {}

            for key, data in pkl_data.items():
                dtype = str(type(data))

                if 'chumpy' in dtype:
                    # Convert chumpy
                    output_data[key] = np.array(data)
                elif 'scipy.sparse' in dtype:
                    # Convert scipy sparse matrix
                    output_data[key] = data.toarray()
                elif 'str' in dtype:
                    output_data[key] = data.encode('ascii', 'replace')
                else:
                    output_data[key] = data

            model_fname = path.split(pkl_path)[1].split(".pkl")[0]

            output_path = path.join(output_dir, model_fname + '.npz')
            np.savez_compressed(output_path, **output_data)

            counter = counter + 1


def main(filepath):
    print("filepath: ", filepath)

    if (type(filepath) == list):
        for f in filepath:
            convert_to_npz(f)
    else:
        convert_to_npz(filepath)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            'Usage: python pkl2npz.py /dir/to/pkl/files'
        )
        exit(-1)
    main(sys.argv[1:])
