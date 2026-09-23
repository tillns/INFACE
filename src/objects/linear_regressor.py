"""
Class that stores the regressor matrix and intercept vector.
Loading and saving is done over json files.
The class contains a method to translate any mesh latent code to the respective output latent code,
as the regressor matrix was trained on.
Training a new linear regressor is also possible via train_linear_regressor().
You can easily use it by calling this script and passing the
paths to the models and paired registered meshes as arguments. Use --help for more information.

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
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from typing import Tuple, Union, List, Dict

from models import get_model_path
from src.objects.mesh import Mesh


class LinearRegressor:
    """
    Very simple class that basically just stores the regressor matrix and intercept vector.
    Loading and saving is done over json files.
    The class contains a method to translate any mesh latent code to the respective output latent code,
    as the regressor matrix was trained on.
    """

    def __init__(self, matrix: np.ndarray, intercept: np.ndarray) -> None:
        """
        :param matrix: nxm regressor weight matrix (numpy).
        :param intercept: n-dimensional numpy vector specifying what's added
                          after multiplying the matrix with some input vector.
        """
        self.regressor_matrix = np.asarray(matrix)
        self.intercept_vector = np.asarray(intercept)


    @classmethod
    def from_json(cls, path_to_json_file: Union[str, Path]) -> "LinearRegressor":
        """
        Class method that loads the regressor matrix and intercept vector
        from the json file and returns a LinearRegressor object instantiated with these two.
        :param path_to_json_file: Path to json file that contains the regressor matrix.
        :return: Matrix as numpy array.
        """
        with open(get_model_path(path_to_json_file), "r") as f:
            data = json.load(f)
        matrix = np.asarray(data["matrix"])
        intercept = np.asarray(data["intercept"])
        return cls(matrix, intercept)

    def translate_latent_code(self, input_latent_code: np.ndarray) -> np.ndarray:
        """
        Translate the latent code of a mesh to another latent code, as the regressor matrix was trained.
        :param input_latent_code: Numpy matrix of the latent code of the mesh within the source 3DMM space.
        :return Numpy array that specifies the latent code of the translated mesh within the target 3DMM space.
        """
        return self.regressor_matrix.dot(input_latent_code) + self.intercept_vector

    def save_to_json(self, path_to_json_file: Union[str, Path]) -> None:
        """
        Save linear regressor (the regressor matrix) to a json file. The same json file can be used to load
        the regressor again.
        :param path_to_json_file: Path to json file into which the regressor matrix and intercept vector are saved
                                 (an existing path is overwritten!)
        """
        data_json = {"matrix": self.regressor_matrix.tolist(), "intercept": self.intercept_vector.tolist()}
        with open(path_to_json_file, "w") as f:
            json.dump(data_json, f)

    @staticmethod
    def train_linear_regressor(
            first_model_path: Union[str, Path], second_model_path: Union[str, Path],
            first_meshes_or_paths: List[Union[Mesh, str, Path]],
            second_meshes_or_paths: List[Union[Mesh, str, Path]]) -> Dict[str, "LinearRegressor"]:
        """
        Train a linear regressor to translate from the first to the second model and vice versa.
        :param first_model_path: Path to the first morphable model.
        :param second_model_path: Path to the second morphable model.
        :param first_meshes_or_paths: List of sample registered meshes (or paths to them) belonging to the first model.
        :param second_meshes_or_paths: List of sample registered meshes (or paths to them)
                                       belonging to the second model.
        :return: Dictionary with two entries:
            "forth": LinearRegressor trained to predict latent of second model from latent of first model.
            "back": LinearRegressor trained to predict latent of first model from latent of second model.
        """
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import KFold
        from src.objects.morphable_model import MorphableModel

        #
        # We first compute the latent codes from the meshes
        first_model = MorphableModel.load_correct_morphable_model(first_model_path)
        second_model = MorphableModel.load_correct_morphable_model(second_model_path)

        def get_model_data(current_model: MorphableModel, current_meshes_or_paths):
            """
            For the current model, compute the latent codes for the provided meshes, and the respective
            reconstructed vertices.
            """
            latent_codes, reconstructed_vertices = [], []
            for current_mesh_or_path in current_meshes_or_paths:
                current_mesh = current_mesh_or_path if isinstance(current_mesh_or_path, Mesh) else Mesh.load(
                    current_mesh_or_path)
                latent_code = current_model.encode(current_mesh.get_vertices())
                latent_codes.append(latent_code)
                reconstructed_vertices_ind = current_model.decode(latent_code)
                reconstructed_vertices.append(reconstructed_vertices_ind)
            return np.asarray(latent_codes), np.asarray(reconstructed_vertices)

        # These are our training variables
        first_latent_codes, first_reconstructed_vertices = get_model_data(first_model, first_meshes_or_paths)
        second_latent_codes, second_reconstructed_vertices = get_model_data(second_model, second_meshes_or_paths)

        # 5-fold cross validation to find best regularization setting
        kf = KFold(n_splits=5, shuffle=False)

        # We compute a forward and a backward regressor, so one predicting second latents from first,
        # and the other vice versa
        regressors = {}
        for model_direction in ["forth", "back"]:
            # Define source and target
            if model_direction == "forth":
                X = first_latent_codes
                Y = second_latent_codes
                Y_recon = second_reconstructed_vertices
                Y_model = second_model
            else:
                X = second_latent_codes
                Y = first_latent_codes
                Y_recon = first_reconstructed_vertices
                Y_model = first_model
            errors_per_alpha = {}
            # We iterate over several regularization settings and for each we do k-fold cross validation.
            # We think regularization is quite important in this setting, since it's an n to m mapping,
            # so the regressor could easily overfit even on hundreds of paired samples.
            for alpha in [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]:
                mse_scores = []
                for train_index, test_index in kf.split(X):
                    # Train regressor to predict the latent codes in Y from the latent codes in X
                    X_train, X_test = X[train_index], X[test_index]
                    Y_train, Y_recon_test = Y[train_index], Y_recon[test_index]
                    regressor = Ridge(alpha=alpha, fit_intercept=True)
                    regressor.fit(X_train, Y_train)

                    # We evaluate the test accuracy not via the latent fitting error, but
                    # via the actual vertex reconstruction from the predicted vs ground truth
                    # test latents
                    Y_pred = regressor.predict(X_test)
                    Y_pred_recon = Y_model.decode(Y_pred)
                    mse = np.linalg.norm(Y_recon_test - Y_pred_recon, axis=-1)
                    mse_scores.extend(mse)
                # We keep track of the vertex reconstruction errors for each alpha
                errors_per_alpha[alpha] = (np.mean(mse_scores), np.std(mse_scores))

            # We choose the best alpha based on the minimum average plus stdev
            best_alpha = min(errors_per_alpha, key=lambda p: errors_per_alpha[p][0]+errors_per_alpha[p][1])

            # We retrain the final model on the whole dataset with the best alpha
            final_regressor = Ridge(alpha=best_alpha, fit_intercept=True)
            final_regressor.fit(X, Y)
            regressors[model_direction] = LinearRegressor(
                matrix=final_regressor.coef_, intercept=final_regressor.intercept_)
        return regressors


# The main method allows for training new linear regressors between two models given paired samples
def main():
    parser = ArgumentParser()
    parser.add_argument('--first_model_path', type=str, required=True,
                        help="Absolute path to autoencoder or PCA model file with .h5 ending.")
    parser.add_argument('--second_model_path', type=str, required=True,
                        help="Absolute path to second autoencoder or PCA model to translate "
                             "to and from via linear regressor.")
    parser.add_argument('--first_mesh_folders', type=str, required=True, nargs="+",
                        help="Provide one or more folders containing registered meshes belonging to the first model.")
    parser.add_argument('--second_mesh_folders', type=str, required=True, nargs="+",
                        help="Provide one or more folders containing registered meshes belonging to the second model.")
    args = parser.parse_args()

    # Load the mesh paths from the folders
    first_mesh_paths = Mesh.get_all_mesh_files_in_path(args.first_mesh_folders)
    second_mesh_paths = Mesh.get_all_mesh_files_in_path(args.second_mesh_folders)

    if not len(first_mesh_paths) == len(second_mesh_paths):
        raise ImportError("Couldn't find the same number of meshes in first and second folder(s).")

    # Load the model paths and train the regressors
    first_model_path = Path(args.first_model_path)
    second_model_path = Path(args.second_model_path)
    regressors = LinearRegressor.train_linear_regressor(
        first_model_path=args.first_model_path, second_model_path=args.second_model_path,
        first_meshes_or_paths=first_mesh_paths, second_meshes_or_paths=second_mesh_paths
    )

    # Save the regressors in the directories of the morphable model files indicating the direction in the name.
    regressors["forth"].save_to_json(
        first_model_path.with_name(f"regressor_{first_model_path.stem}_to_{second_model_path.stem}.json"))
    regressors["back"].save_to_json(
        second_model_path.with_name(f"regressor_{second_model_path.stem}_to_{first_model_path.stem}.json"))


if __name__ == '__main__':
    main()