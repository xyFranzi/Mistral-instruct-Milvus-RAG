#!/usr/bin/env python3
"""
Simple test script to verify the GPT evaluation setup
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import OPENAI_API_KEY
    print("✓ Successfully imported OpenAI API key from config")
    print(f"  API key starts with: {OPENAI_API_KEY[:10]}...")
except ImportError as e:
    print(f"✗ Failed to import config: {e}")
    sys.exit(1)

try:
    import openai
    print("✓ OpenAI library imported successfully")
except ImportError:
    print("✗ OpenAI library not found. Install with: pip install openai==0.28.1")
    sys.exit(1)

# Check test results directory
test_results_dir = Path(__file__).parent.parent / "test_results"
if test_results_dir.exists():
    json_files = list(test_results_dir.glob("*.json"))
    print(f"✓ Found {len(json_files)} JSON files in test_results directory")
    for file in json_files:
        print(f"  - {file.name}")
else:
    print("✗ test_results directory not found")

# Check eval results directory
eval_results_dir = Path(__file__).parent.parent / "eval_results"
if eval_results_dir.exists():
    print("✓ eval_results directory exists")
else:
    print("✗ eval_results directory not found")

print("\nSetup verification complete!")
print("\nTo run the evaluation:")
print("1. Make sure you have internet connection for OpenAI API")
print("2. Run: python gpt_eval.py")
