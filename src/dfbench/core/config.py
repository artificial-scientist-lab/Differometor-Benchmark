import argparse
from typing import Any


def create_parser(
    params: dict[str, Any], description: str = "Optimization configuration"
) -> argparse.ArgumentParser:
    """
    Create an argument parser from a dictionary of parameter names and default values.

    Args:
        params: Dictionary mapping parameter names to default values
        description: Description for the argument parser

    Returns:
        Configured ArgumentParser

    Example:
        >>> parser = create_parser({
        ...     "pop_size": 100,
        ...     "generations": 10,
        ...     "learning_rate": 0.01,
        ...     "use_cuda": False
        ... })
        >>> args = parser.parse_args()
    """
    parser = argparse.ArgumentParser(description=description)

    for param_name, default_value in params.items():
        # Convert snake_case to kebab-case for CLI
        arg_name = f"--{param_name.replace('_', '-')}"

        # Infer type from default value
        param_type = type(default_value)

        # Handle booleans as flags
        if isinstance(default_value, bool):
            parser.add_argument(
                arg_name,
                action="store_true" if not default_value else "store_false",
                help=f"{param_name} (default: {default_value})",
            )
        else:
            parser.add_argument(
                arg_name,
                type=param_type,
                default=default_value,
                help=f"{param_name} (default: {default_value})",
            )

    return parser
