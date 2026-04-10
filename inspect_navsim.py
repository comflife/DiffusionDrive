"""Quick inspection of navsim pkl data format."""
import pickle
import sys

pkl_path = '/data/navsim/dataset/navsim_logs/trainval/2021.05.12.19.36.12_veh-35_01744_01934.pkl'
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print(f"Type: {type(data)}")
print(f"Length: {len(data)}")

if isinstance(data, list):
    print(f"\nFrame type: {type(data[0])}")
    print(f"Frame keys: {list(data[0].keys())}")
    print(f"\nego2global_translation: {data[0].get('ego2global_translation')}")
    print(f"ego2global_rotation: {data[0].get('ego2global_rotation')}")
    print(f"ego_dynamic_state: {data[0].get('ego_dynamic_state')}")
    print(f"token: {data[0].get('token')}")
    print(f"roadblock_ids type: {type(data[0].get('roadblock_ids'))}")
    
    # Print first 5 frames' ego poses
    print("\n--- First 5 frames ego poses ---")
    for i in range(min(5, len(data))):
        t = data[i].get('ego2global_translation', 'N/A')
        r = data[i].get('ego2global_rotation', 'N/A')
        print(f"Frame {i}: trans={t}, rot={r}")
