#!/usr/bin/env python3
import os
import argparse
from EMBER import create_vectorized_features

# Check if the input is a valid directory
def dir_path(string):
   if os.path.isdir(string):
      return string
   else:
      raise NotADirectoryError(string)


if __name__ == "__main__":

   # Parse command-line arguments
   parser = argparse.ArgumentParser(description="Generate EMBER feature vectors for MOTIF")
   parser.add_argument("dataset_dir", type=dir_path, help="Path to directory containing MOTIF dataset")
   args = parser.parse_args()
   create_vectorized_features(args.dataset_dir)
