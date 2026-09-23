from pathlib import Path
from typing import Union
from src import utils

models_folder = Path(__file__).parent

def get_model_path(model_name_or_path: Union[str, Path]) -> Path:
    model_path = Path(model_name_or_path)
    if not model_path.is_absolute():
        model_path = models_folder.joinpath(model_path)
        if not model_path.exists():
            possible_paths = utils.get_all_files_in_path(models_folder, regex_names=model_path.name)
            if len(possible_paths) == 1:
                model_path = possible_paths[0]
    if not model_path.exists():
        raise FileNotFoundError(f"Couldn't locate model file: {model_name_or_path}")
    return model_path