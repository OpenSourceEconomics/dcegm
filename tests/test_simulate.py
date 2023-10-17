from pathlib import Path

import pandas as pd
from jax.config import config

config.update("jax_enable_x64", True)

# Obtain the test directory of the package.
TEST_DIR = Path(__file__).parent

# Directory with additional resources for the testing harness
TEST_RESOURCES_DIR = TEST_DIR / "resources"

TEST_DF = pd.read_csv(TEST_RESOURCES_DIR / "df_simulate_fedor.csv")
