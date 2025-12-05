#!/usr/bin/env python3
import json
import argparse
import sys

def trans(new_to_origin, input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        run_dict = json.load(f)

    new_run_dict = {}
    for case_id, node_scores in run_dict.items():
        for node_id, score in node_scores.items():
            origin_id = new_to_origin.get(node_id)
            if origin_id is None:
                print(f"Warning: node_id '{node_id}' not found in mapping. Skipping.", file=sys.stderr)
                continue
            if case_id not in new_run_dict:
                new_run_dict[case_id] = {}
            new_run_dict[case_id][origin_id] = score

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_run_dict, f, indent=2, ensure_ascii=False)

    return new_run_dict

def main():
    parser = argparse.ArgumentParser(description="Transform run results using a mapping from new IDs to original IDs.")
    parser.add_argument("--mapping", required=True, help="Path to the mapping JSON file (new_id -> origin_id)")
    parser.add_argument("--run", required=True, help="Path to the input run JSON file (case_id -> {new_id: score})")
    parser.add_argument("--result", required=True, help="Path to the output result JSON file")

    args = parser.parse_args()

    with open(args.mapping, 'r', encoding='utf-8') as f:
        new_to_origin = json.load(f)

    trans(new_to_origin, args.run, args.result)

if __name__ == "__main__":
    main()