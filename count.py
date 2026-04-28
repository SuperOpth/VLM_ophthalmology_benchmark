import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task_type", type=str, required=True, choices=["management", "diagnosis"], help="Task type")
parser.add_argument("--multimodal", default=True, action="store_true", help="Whether to use multimodal input")
parser.add_argument("--test_file", type=str, default="data/test.jsonl", help="Path to the test file")
args = parser.parse_args()

MULTIMODAL = args.multimodal        # True: 传入figure / False: 不传入
TASK_TYPE = args.task_type  # "management" / "diagnosis"

models = [
    "google/gemini-2.0-flash-001",
    "openai/gpt-5-nano",
    "openai/gpt-5.2",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "google/gemini-3-flash-preview",
    "qwen/qwen3-vl-30b-a3b-instruct",
    "qwen/qwen3-vl-235b-a22b-instruct",
    "x-ai/grok-4.1-fast"
]

for model in models:
    output_file = f"result/{model.split('/')[-1]}_{TASK_TYPE}_{'multimodal' if MULTIMODAL else ''}.json"

    test_file = args.test_file

    results = []
    correct = 0
    total = 0

    try:
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                records = json.load(f)
                results.extend(records)
            
            if TASK_TYPE == "management":
                for record_id, record in enumerate(results, start=1):
                    if record.get("predicted_answer","") == "":
                        # print(record_id, "has no answer, skipping...")
                        continue
                    correct += 1 if record.get("predicted_answer","") == record.get("correct_answer","") else 0
                    total += 1
    except:
        print("No previous results found, starting fresh.")

    print("Model:", model,
        "Correct:", correct
        , "Total:", total
        , "Accuracy:", f"{(correct/total*100):.2f}%" if total > 0 else "N/A"
        )

