"""
Very simple class that basically just stores the regressor matrix and intercept vector.
Loading and saving is done over json files.
The class contains a method to translate any mesh latent code to the respective output latent code,
as the regressor matrix was trained on.

Author: Till Schnabel (contact till.schnabel@inf.ethz.ch)

MIT License

Copyright (c) 2025 ETH Zurich, Till Schnabel

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
import json
from argparse import ArgumentParser

import numpy as np
from typing import Tuple



class LinearRegressor:
    """
    Very simple class that basically just stores the regressor matrix and intercept vector.
    Loading and saving is done over json files.
    The class contains a method to translate any mesh latent code to the respective output latent code,
    as the regressor matrix was trained on.
    """

    def __init__(self, path_to_json_file: str) -> None:
        """
        :param path_to_json_file: Provide the path to the json file
        that contains the regressor matrix and intercept vector.
        """
        self.regressor_matrix, self.intercept_vector = self.load_linear_regressor_from_json(path_to_json_file)


    @staticmethod
    def load_linear_regressor_from_json(path_to_json_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Static method that loads the regressor matrix and intercept vector
        from the json file and returns the two as numpy arrays.
        :param path_to_json_file: Path to json file that contains the regressor matrix.
        :return: Matrix as numpy array.
        """
        with open(path_to_json_file, "r") as f:
            data = json.load(f)
        matrix = np.asarray(data["matrix"])
        intercept = np.asarray(data["intercept"])
        return matrix, intercept

    def translate_latent_code(self, input_latent_code: np.ndarray) -> np.ndarray:
        """
        Translate the latent code of a mesh to another latent code, as the regressor matrix was trained.
        :param input_latent_code: Numpy matrix of the latent code of the mesh within the source 3DMM space.
        :return Numpy array that specifies the latent code of the translated mesh within the target 3DMM space.
        """
        return self.regressor_matrix.dot(input_latent_code) + self.intercept_vector

    def save_linear_regressor(self, path_to_json_file: str) -> None:
        """
        Save linear regressor (the regressor matrix) to a json file. The same json file can be used to load
        the regressor again.
        :param path_to_json_file: Path to json file into which the regressor matrix and intercept vector are saved
                                 (an existing path is overwritten!)
        """
        data_json = {"matrix": self.regressor_matrix.tolist(), "intercept": self.intercept_vector.tolist()}
        with open(path_to_json_file, "w") as f:
            json.dump(data_json, f)

def main():
    from morphable_model import MorphableModel
    parser = ArgumentParser()
    parser.add_argument('--first_model_path', type=str, required=True,
                        help="Absolute path to autoencoder or PCA model file with .h5 ending.")
    parser.add_argument('--second_model_path', type=str, required=True,
                        help="Absolute path to second autoencoder or PCA model to translate to via linear regressor.")
    parser.add_argument('--path_to_linear_regressor', type=str, required=True,
                        help="Absolute path to linear regressor file with .json ending. Needs to be combined with ")

    args = parser.parse_args()
    first_model = MorphableModel.load_correct_morphable_model(args.first_model_path)
    second_model = MorphableModel.load_correct_morphable_model(args.second_model_path)
    linear_regressor = LinearRegressor(args.path_to_linear_regressor)
    sample_meshes = first_model.sample_meshes(5)
    for sample_mesh in sample_meshes:
        translated_sample = first_model.convert_mesh_with_linear_regressor(linear_regressor, sample_mesh, second_model)
        sample_mesh.show_multiple_meshes(sample_mesh, translated_sample)


if __name__ == '__main__':
    main()