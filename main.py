# Oct15
import os
import json
import re
import time
import random
from pydantic import Field
from openai import OpenAI
from alive_progress import alive_it

import atexit
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, default="google/gemini-2.0-flash-001", help="Model name")
parser.add_argument("--task_type", type=str, required=True, choices=["management", "diagnosis", "judge"], help="Task type")
parser.add_argument("--multimodal", action="store_true", help="Whether to use multimodal input")
parser.add_argument("--test_file", type=str, default="data/test.jsonl", help="Path to the test file")
parser.add_argument("--reasoning", type=bool, help="Whether to enable reasoning")
args = parser.parse_args()

MULTIMODAL = args.multimodal        # True: 传入figure / False: 不传入
TASK_TYPE = args.task_type  # "management" / "diagnosis"
MODEL = args.model if TASK_TYPE != 'judge' else "google/gemini-3-flash-preview"
REASONING = args.reasoning


OPEN_ROUTER_API_KEY = os.environ['OPEN_ROUTER_API_KEY']

OPEN_ROUTER_BASE_URL = "https://openrouter.ai/api/v1"

output_file = f"result/{args.model.split('/')[-1]}_{'r' if REASONING else 'nr'}_{TASK_TYPE if TASK_TYPE != 'judge' else 'diagnosis'}_{'multimodal' if MULTIMODAL else ''}.json"

test_file = args.test_file if TASK_TYPE != "judge" else output_file

# OPENAI兼容
class Agent:
    api_key: str = Field(...)
    model: str = Field(default=MODEL)
    base_url: str = Field(default=None)

    def __init__(self, api_key: str, model: str, base_url: str):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
    def chat(self, messages):
        try:
            resp = self.client.chat.completions.create(
                model=self.model, 
                messages=messages,
                extra_body= {"reasoning":{"enabled": REASONING if TASK_TYPE != 'judge' else False}}
                )
        except Exception as e:
            print(f"ERROR: {e}")
            return "" 
        return resp

llm = Agent(api_key=OPEN_ROUTER_API_KEY, model=MODEL, base_url=OPEN_ROUTER_BASE_URL)

# 提取答案选项
def extract_first_capital_letter(text: str):
    if not text:
        return ""
    m = re.search(r'\b([A-Z])\b', text)
    if m:
        return m.group(1)
    m = re.search(r'([A-Z])', text)
    return m.group(1) if m else ""

total = 0
correct = 0
results = []

try:
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            records = json.load(f)
            results.extend(records)
        
        if TASK_TYPE == "management":
            for record in results:
                if record.get("predicted_answer","") == "":
                    continue
                correct += 1 if record.get("predicted_answer","") == record.get("correct_answer","") else 0
                total += 1
except:
    print("No previous results found, starting fresh.")

def onExit():
    print("\nSaving intermediate results...")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(results, ensure_ascii=False, indent=4))

atexit.register(onExit)

def diagnose(record_id, data):
    st = time.time()
    case = data.get("case", "")
    question_instruction = data.get("instruction","You are an ophthalmologist, what is the most likely diagnosis for this patient?")
    ans = str(data.get("answer_idx", "")).strip().upper()
    real_diag = data.get("diagnosis", "")

    # 图片处理URL
    figure_b64 = data.get("figure") or None

    if figure_b64 == None and MULTIMODAL:
        print(f"Record {record_id} has no figure, skipping.")
        return {}

    legend = data.get("figure_legend")
    if not legend or str(legend).lower() in ["nan", "none", "null"]:
        legend = None
    img_data_url = f"data:image/png;base64,{figure_b64}" if MULTIMODAL and figure_b64 else None

    prompt_text = "\n\n".join([
        "Case: ", case,
        f"Figure legend: {legend}" if legend else "",
        question_instruction,
    ])
        
    message_content = []

    instruction_text = prompt_text + "\nINSTRUCTIONS: " + (
        "Provide ONLY the final answer as one concise diagnosis. No explanation."
    )
    message_content.append({"type":"text","text": instruction_text})

    if img_data_url:
        message_content.append({"type":"image_url","image_url": img_data_url})

    messages = [{"role":"user","content": message_content}]

    resp = llm.chat(messages=messages)
    predicted_text = resp.choices[0].message.content.strip()
    total_tokens = resp.usage.total_tokens
    prompt_tokens = resp.usage.prompt_tokens
    completion_tokens = resp.usage.completion_tokens

    diag = predicted_text

    result_record = {
        "record_id": record_id,
        "task_type": TASK_TYPE,
        "model": MODEL,
        "correct_diagnosis": real_diag,
        "predicted_diagnosis": diag,
        "time_cost": time.time() - st,
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
        
    return result_record

def judge(data):
    correct_diagnosis = data.get("correct_diagnosis", "").strip().lower()
    predicted_diagnosis = data.get("predicted_diagnosis", "").strip().lower()

    if correct_diagnosis == "" or predicted_diagnosis == "":
        return 0
    
    if data.get("score", None) is not None:
        return data
    
    prompt = f"""You are a medical doctor with extensive clinical experience, proficient in diagnostic terminology and clinical significance across various diseases. Please evaluate the match between the predicted diagnosis and the correct diagnosis using the following 5-point scale:

    【5-Point Scoring Criteria】
    5 points (Exact Match): The predicted diagnosis is identical in terminology to the correct diagnosis.
    4 points (Clinically Equivalent): Different terminology but completely identical in clinical meaning (e.g., abbreviations vs full names, synonyms).
    3 points (Strongly Related): Belong to closely related diseases (e.g., different lesions of the same organ, etiology and direct complications).
    2 points (Weakly Related): Belong to the same general category but with low relevance (e.g., non-directly related diseases in the same system).
    1 point (Incorrect): No clinical relevance between the predicted and correct diagnoses.

    【Task Requirements】
    1. Assign a score (1-5) based on the above criteria.
    2. Provide a brief reason (max 30 words) explaining the scoring basis.
    3. Output in JSON format containing only two fields: "score" and "reason".
    4. Return ONLY the JSON, with no additional text. Example:
    {{
        "score": 4,
        "reason": "Different terminology but clinically equivalent."
    }}

    【Current Case】
    Correct diagnosis: {correct_diagnosis}
    Predicted diagnosis: {predicted_diagnosis}
    """

    messages = [
        {"role":"user","content": prompt}
    ]
    resp = llm.chat(messages=messages)
    response = json.loads(resp.choices[0].message.content.strip())

    new_data = data.copy()
    new_data["score"] = response.get("score", 0)
    new_data["reason"] = response.get("reason", "")

    return new_data

def shuffle_options(options, ans):
    keys = list(options.keys())
    random.shuffle(keys)
    shuffled_options = {chr(65+idx): options[key] for idx, key in enumerate(keys)}
    
    shuffled_ans = ""

    for key in keys:
        if key == ans:
            shuffled_ans = chr(65 + keys.index(key))
            break

    return shuffled_options, shuffled_ans

def manage(record_id, data):
    st = time.time()
    case = data.get("case", "")
    question_instruction = data.get("instruction","You are an ophthalmology attending. What is the most appropriate next step for this patient?")
    options = data.get("options", {}) or {}
    ans = str(data.get("answer_idx", "")).strip().upper()

    shuffled_options, shuffled_ans = shuffle_options(options, ans)
    options_text = "\n".join([f"{key}: {value}" for key, value in shuffled_options.items()])

    # 图片处理URL
    figure_b64 = data.get("figure") or None

    if figure_b64 == None and MULTIMODAL:
        print(f"Record {record_id} has no figure, skipping.")
        return {}

    legend = data.get("figure_legend")
    if not legend or str(legend).lower() in ["nan", "none", "null"]:
        legend = None
    img_data_url = f"data:image/png;base64,{figure_b64}" if MULTIMODAL and figure_b64 else None

    prompt_text = "\n\n".join([
        "Case: ", case,
        f"Options: {options_text}" if TASK_TYPE=="management" else "",
        f"Figure legend: {legend}" if legend else "",
        question_instruction,
    ])
        
    message_content = []

    instruction_text = prompt_text + "\nINSTRUCTIONS: " + (
        "Provide ONLY the final answer as a single capital letter (A, B, C, or D). No explanation."
    )
    message_content.append({"type":"text","text": instruction_text})

    if img_data_url:
        message_content.append({"type":"image_url","image_url": img_data_url})

    messages = [{"role":"user","content": message_content}]

    resp = llm.chat(messages=messages)
    predicted_text = resp.choices[0].message.content.strip()
    total_tokens = resp.usage.total_tokens
    prompt_tokens = resp.usage.prompt_tokens
    completion_tokens = resp.usage.completion_tokens

    predicted = extract_first_capital_letter(predicted_text)

    result_record = {
        "record_id": record_id,
        "task_type": TASK_TYPE,
        "model": MODEL,
        "shuffle_options": shuffled_options,
        "correct_answer": shuffled_ans,
        "predicted_answer": predicted,
        "time_cost": time.time() - st,
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
        
    return result_record

print(test_file)
with open(test_file, "r", encoding="utf-8") as f:
    dataset = alive_it(enumerate(f if TASK_TYPE != 'judge' else json.load(f), start=1))
    for record_id, line in dataset:
        if record_id <= len(results) and results[record_id-1].get('record_id', -1) == record_id and (
            (TASK_TYPE == "management" and results[record_id-1].get("predicted_answer","") != "") or
            (TASK_TYPE == "diagnosis" and results[record_id-1].get("predicted_diagnosis","") != "") or
            (TASK_TYPE == "judge" and results[record_id-1].get("score", None) is not None)
        ):
            continue
        
        data = json.loads(line.strip()) if TASK_TYPE != 'judge' else line

        if TASK_TYPE == "diagnosis":
            result = diagnose(record_id, data)
        elif TASK_TYPE == "management":
            result = manage(record_id, data)
            correct += 1 if result.get("predicted_answer","") == result.get("correct_answer","") else 0
        else:
            result = judge(data)
        if result:
            total += 1
        
        if len(results) >= record_id:
            results[record_id-1] = result
        else:
            results.append(result)
        if TASK_TYPE == "management":
            dataset.text(f"Acc {correct}/{total} {correct/total*100:.2f}%")

