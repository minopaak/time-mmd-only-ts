import importlib
from pathlib import Path


def get_model_class(model_name: str):
    models_dir = Path(__file__).resolve().parent
    model_files = {
        p.stem for p in models_dir.glob("*.py")
        if p.stem not in {"__init__", "factory"}
    }

    if model_name not in model_files:
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {sorted(model_files)}"
        )

    module = importlib.import_module(f"models.{model_name}")

    if not hasattr(module, "Model"):
        raise AttributeError(
            f"models.{model_name} must define a class named `Model`"
        )

    return module.Model