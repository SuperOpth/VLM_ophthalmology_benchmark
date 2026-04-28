import argparse
import datetime
import json
import logging
import os
import shlex
import signal
import shutil
import subprocess
import sys
import time
import argparse


SHUTDOWN = False

parser = argparse.ArgumentParser(description="Maintain and restart a long-running command.")
parser.add_argument("--task_type", type=str, required=True, choices=["management", "diagnosis", "judge"], help="Task type")
parser.add_argument("--reasoning", type=bool, help="Enable reasoning")
args = parser.parse_args()


cmd = ["python", "main.py", "--task_type", args.task_type, "--test_file", "data/jocc_449.jsonl","--reasoning", str(args.reasoning)]

r_models = [
   "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    "openai/gpt-5.2",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "google/gemini-3-flash-preview",
    "qwen/qwen3-vl-30b-a3b-thinking",
    "qwen/qwen3-vl-235b-a22b-thinking",
    "x-ai/grok-4.1-fast"
]

nr_models = [
    "google/gemini-2.0-flash-001",
    "google/gemini-2.5-flash",
    "google/gemini-3-flash-preview",
    "qwen/qwen3-vl-30b-a3b-instruct",
    "qwen/qwen3-vl-235b-a22b-instruct",
    "x-ai/grok-4.1-fast"
]

models = r_models if args.reasoning else nr_models

def sigterm_handler(signum, frame):
    global SHUTDOWN
    SHUTDOWN = True

signal.signal(signal.SIGTERM, sigterm_handler)
signal.signal(signal.SIGINT, sigterm_handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

def run_watchdog(cmd, always_restart=False, max_restarts_per_hour=10):
    backoff = 5
    restarts = []
    while not SHUTDOWN:
        now = time.time()
        restarts = [t for t in restarts if now - t < 3600]
        if len(restarts) >= max_restarts_per_hour:
            logging.error("Too many restarts in the last hour (%d), sleeping 5m", len(restarts))
            time.sleep(300)
        logging.info("Starting command: %s", cmd)
        with open("maintained_process.log", "ab") as logfile:
            proc = subprocess.Popen(shlex.split(cmd), stderr=subprocess.STDOUT, env=os.environ)
            while True:
                if SHUTDOWN:
                    logging.info("Shutdown requested, terminating child (pid %d)", proc.pid)
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except Exception:
                        proc.kill()
                    return
                ret = proc.poll()
                if ret is None:
                    time.sleep(1)
                    continue
                # process exited
                logging.info("Process exited with code %s", ret)
                break

        if not always_restart and ret == 0:
            logging.info("Process exited cleanly and --no-restart requested; stopping watchdog.")
            return

        # record restart and exponential backoff
        restarts.append(time.time())
        if SHUTDOWN:
            return
        logging.info("Restarting after %ds...", backoff)
        time.sleep(backoff)
        backoff = min(backoff * 2, 300)

def main():

    for model in models:
        full_cmd = cmd + ["--model", model]
        run_watchdog(" ".join(full_cmd), always_restart=False, max_restarts_per_hour=10)

if __name__ == "__main__":
    main()