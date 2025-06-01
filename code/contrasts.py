import json
import os

class ContrastManager:
    """
    Manages a collection of contrasts, primarily loaded from a file.
    """
    def __init__(self, filepath="contrasts.json"):
        """
        Initializes the ContrastManager by loading contrasts from a specified JSON file.

        Parameters:
        filepath (str): The path to the JSON file containing contrast definitions.
        """
        self.contrasts = {}
        self._load_contrasts_from_file(filepath)

    def _load_contrasts_from_file(self, filepath):
        """
        (Internal) Loads contrast definitions from the specified JSON file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Contrast file not found at: {filepath}")

        try:
            with open(filepath, 'r') as f:
                loaded_data = json.load(f)
                if not isinstance(loaded_data, dict):
                    raise ValueError("Contrast file must contain a top-level dictionary.")
                self.contrasts = loaded_data
            print(f"Successfully loaded {len(self.contrasts)} contrasts from {filepath}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {filepath}: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while loading contrasts: {e}")

    def get_contrast(self, contrast_name):
        """
        Retrieves a specific contrast definition by its name.

        Parameters:
        contrast_name (str): The name of the contrast to retrieve.

        Returns:
        dict: The contrast definition.

        Raises:
        ValueError: If the contrast with the given name is not found.
        """
        if contrast_name in self.contrasts:
            return self.contrasts[contrast_name]
        else:
            raise ValueError(f"Contrast '{contrast_name}' not found.")

    def list_contrasts(self):
        """
        Lists all available contrast names.

        Returns:
        list: A list of all contrast names.
        """
        return list(self.contrasts.keys())
