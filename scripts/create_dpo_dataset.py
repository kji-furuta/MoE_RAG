#!/usr/bin/env python3
"""
DPO Dataset Creator - Convert malformed JSON to proper JSONL format
Handles multi-line JSON objects with LaTeX formulas
"""
import json
import re
from pathlib import Path
from typing import List, Dict


def extract_json_objects(content: str) -> List[Dict]:
    """
    Extract JSON objects from malformed multi-line JSON content.

    Args:
        content: Raw file content with multi-line JSON objects

    Returns:
        List of parsed JSON objects
    """
    objects = []
    lines = content.split('\n')

    current_obj_lines = []
    brace_depth = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Track brace depth
        for char in stripped:
            if char == '{':
                brace_depth += 1
            elif char == '}':
                brace_depth -= 1

        current_obj_lines.append(stripped)

        # When we reach brace_depth == 0, we have a complete object
        if brace_depth == 0 and current_obj_lines:
            obj_str = ' '.join(current_obj_lines)

            try:
                # Try to parse as-is
                obj = json.loads(obj_str)
                objects.append(obj)
            except json.JSONDecodeError:
                # If parsing fails, try to fix common issues
                try:
                    # Remove trailing commas before closing braces
                    obj_str = re.sub(r',(\s*})', r'\1', obj_str)
                    obj = json.loads(obj_str)
                    objects.append(obj)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse object: {e}")
                    print(f"  First 100 chars: {obj_str[:100]}...")

            current_obj_lines = []

    return objects


def create_dpo_jsonl(input_path: str, output_path: str):
    """
    Create proper DPO JSONL file from malformed JSON input.

    Args:
        input_path: Path to input file with malformed JSON
        output_path: Path to output JSONL file
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    print(f"Reading from: {input_file}")

    # Read entire file
    with input_file.open('r', encoding='utf-8') as f:
        content = f.read()

    print(f"File size: {len(content)} bytes")

    # Extract JSON objects
    print("Extracting JSON objects...")
    objects = extract_json_objects(content)
    print(f"Extracted {len(objects)} objects")

    # Validate and write to JSONL
    valid_count = 0
    skipped_count = 0

    with output_file.open('w', encoding='utf-8') as f:
        for i, obj in enumerate(objects):
            try:
                # Validate required fields
                if not all(key in obj for key in ['prompt', 'chosen', 'rejected']):
                    print(f"Warning: Object {i} missing required fields: {list(obj.keys())}")
                    skipped_count += 1
                    continue

                # Create DPO format object (only 3 required fields)
                dpo_obj = {
                    'prompt': str(obj['prompt']),
                    'chosen': str(obj['chosen']),
                    'rejected': str(obj['rejected'])
                }

                # Write as single-line JSON
                json.dump(dpo_obj, f, ensure_ascii=False)
                f.write('\n')
                valid_count += 1

            except Exception as e:
                print(f"Warning: Error processing object {i}: {e}")
                skipped_count += 1

    print(f"\n✅ Successfully wrote {valid_count} records to {output_file}")
    if skipped_count > 0:
        print(f"⚠️  Skipped {skipped_count} invalid records")

    # Validation
    print("\n=== Validation ===")
    with output_file.open('r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
        print(f"Total lines in output: {len(lines)}")

        if lines:
            # Show first 3 samples
            for i in range(min(3, len(lines))):
                obj = json.loads(lines[i])
                print(f"\nSample {i+1}:")
                print(f"  Fields: {list(obj.keys())}")
                print(f"  Prompt: {obj['prompt'][:60]}...")
                print(f"  Chosen: {obj['chosen'][:60]}...")
                print(f"  Rejected: {obj['rejected'][:60]}...")

    return valid_count


def main():
    """Main execution"""
    source = "/workspace/data/dpo/request_dpo.jsonl"
    target = "/workspace/data/dpo/preference_dataset.jsonl"

    print("=" * 60)
    print("DPO Dataset Creator")
    print("=" * 60)

    try:
        count = create_dpo_jsonl(source, target)

        print("\n" + "=" * 60)
        print(f"✅ Conversion complete!")
        print(f"   {count} records ready for DPO training")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
