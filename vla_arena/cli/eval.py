import os
import importlib
import importlib.util

def eval_main(args):
    model = args.model
    # Ensure config is an absolute path for easy reading by subprocesses
    config_path = os.path.abspath(str(args.config))

    # 1. Dynamically get the physical path of the corresponding model evaluator.py file
    try:
        module_name = f"vla_arena.models.{model}.evaluator"
        module_spec = importlib.util.find_spec(module_name)
        if module_spec is None or module_spec.origin is None:
            raise ImportError(f"Cannot find module {module_name}")
        
    except ImportError as e:
        raise RuntimeError(
            f"Model '{model}' is not installed or evaluator script not found.\n"
            f"Try: pip install vla-arena[{model}]"
        ) from e

    # 2. Directly import the module and execute main
    module = importlib.import_module(module_name)
    # Pass config path string here, evaluator.py's main function will handle it
    module.main(cfg=config_path)
