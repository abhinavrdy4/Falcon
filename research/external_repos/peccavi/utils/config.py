"""
Official Implementation of ConfigLoader class.

This module provides a utility for loading and accessing configuration parameters 
from YAML files, enabling dynamic attribute assignment for easier access to 
hyperparameters and settings.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import yaml
import torch
import re

# ==============================================================================
# CONFIGURATION MANAGEMENT
# ==============================================================================

class ConfigLoader:
    """
    A class to load configuration settings from a YAML file.

    The class parses a YAML file and dynamically maps its top-level keys 
    as instance attributes, allowing for dot-notation access (e.g., config.learning_rate).

    Attributes:
        config (dict): The raw dictionary containing all loaded configuration data.
        [Dynamic Attributes]: Every key in the YAML file becomes an attribute on this object.
    """

    def __init__(self, config_path):
        """
        Initializes the ConfigLoader and populates attributes from the YAML file.

        Args:
            config_path (str): The system path to the YAML configuration file.

        Side Effects:
            - Opens and reads a file from the disk.
            - Calls `setattr` for every key-value pair found in the YAML root.

        Notes:
            - Uses `yaml.safe_load` for secure parsing.
            - Existing attributes may be overwritten if the YAML contains reserved names.
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Flatten the dictionary into instance attributes for easy access
        for key, value in self.config.items():
            setattr(self, key, value)

        # PECCAVI-FIX 5: device was hard-coded to cuda:1 in config.yaml; allow an override.
        if os.environ.get("PECCAVI_DEVICE"):
            self.device = os.environ["PECCAVI_DEVICE"]
    
    def wandb_config(self):
        """
        Reloads a specific 'config.yaml' file for use with experiment trackers.

        This method specifically targets a file named 'config.yaml' in the 
        current working directory, regardless of the path provided during initialization.

        Returns:
            dict: The dictionary representation of the 'config.yaml' file content.

        Notes:
            - This method does not update the instance attributes of the class.
            - It assumes 'config.yaml' exists in the execution context's root.
        """
        with open('config.yaml') as config_file:
            config_dict = yaml.safe_load(config_file)
        return config_dict

# ==============================================================================
# EXAMPLE USAGE (COMMENTED OUT)
# ==============================================================================

# if __name__ == "__main__":
#     # Usage: Provide path to a YAML file to initialize attributes
#     config = ConfigLoader("../config.yaml")