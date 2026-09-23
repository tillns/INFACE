"""
This script basically just calls the train() method from the PCAMorphableModel class.

Author: Till Schnabel (contact till.schnabel@inf.ethz.ch)

MIT License

Copyright (c) 2026 ETH Zurich, Till Schnabel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# library imports
from argparse import ArgumentParser
# own imports
from src.objects.PCA_morphable_model import PCAMorphableModel

def main():
    parser = ArgumentParser()
    parser.add_argument('--registered_mesh_folders', type=str, nargs="+", required=True,
                        help="Provide one or more folder paths to load the registered meshes from.")
    parser.add_argument('--output_path', type=str, required=True,
                        help="Path to save model under (.h5).")
    parser.add_argument('--mesh_regex_filter', type=str, nargs="+", default=None,
                        help="Optionally filter the meshes to train the model on with this argument.")
    parser.add_argument('--exclude_mesh_regex_filter', type=str, nargs="+", default=None,
                        help="Optionally use this argument to filter meshes to be excluded from training.")
    parser.add_argument('--retain_components', type=float, default=None,
                        help="Optionally specify variation percentage (between 0 and 1),"
                             "or number of components (>=1) to keep.")
    parser.add_argument('--include_colors', action="store_true", default=False,
                        help="Choose this to include average mesh colors into morphable model.")
    args = parser.parse_args()

    PCAMorphableModel.train(
        args.registered_mesh_folders, path_to_hdf5_file=args.output_path,
        mesh_regex_filter=args.mesh_regex_filter, exclude_mesh_regex_filter=args.exclude_mesh_regex_filter,
        retain_components=args.retain_components, include_colors=args.include_colors)


if __name__ == '__main__':
    main()
