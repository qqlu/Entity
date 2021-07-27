import os
import argparse

import torch

def parse_args():
    parser = argparse.ArgumentParser("D2 model converter")
    parser.add_argument("--source_model", default="", type=str, help="Path or url to the  model to convert")
    parser.add_argument("--output_model", default="", type=str, help="Path where to save the converted model")
    return parser.parse_args()

def main():
    args = parse_args()
    
    source_weights = torch.load(args.source_model)
    converted_weights = {}
    keys = list(source_weights.keys())
    
    prefix = 'backbone.bottom_up.'
    for key in keys:
        converted_weights[prefix+key] = source_weights[key]

    torch.save(converted_weights, args.output_model)

if __name__ == "__main__":
    main()