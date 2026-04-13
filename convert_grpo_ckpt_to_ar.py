#!/usr/bin/env python3
"""
Convert GRPO checkpoint to AR-compatible checkpoint.
"""
import argparse
import torch

def convert_checkpoint(input_path: str, output_path: str):
    ckpt = torch.load(input_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        # Convert policy_model.* -> _transfuser_model.*
        if k.startswith("policy_model."):
            new_key = "_transfuser_model." + k[len("policy_model."):]
            new_state_dict[new_key] = v
        # Skip reference_model (not needed for inference)
        elif k.startswith("reference_model."):
            continue
        else:
            new_state_dict[k] = v
    
    # Save with same structure
    new_ckpt = {
        "state_dict": new_state_dict,
    }
    # Copy other metadata if exists
    for key in ["epoch", "global_step", "pytorch-lightning_version", "loops", "optimizer_states", "lr_schedulers", "callbacks"]:
        if key in ckpt:
            new_ckpt[key] = ckpt[key]
    
    torch.save(new_ckpt, output_path)
    print(f"Converted {len(new_state_dict)} keys")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="GRPO checkpoint path")
    parser.add_argument("--output", required=True, help="Output AR-compatible checkpoint path")
    args = parser.parse_args()
    convert_checkpoint(args.input, args.output)
