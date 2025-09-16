
import subprocess, sys

def run(cmd):
    print("+", " ".join(cmd)); subprocess.check_call(cmd)

def test_cli_smoke():
    run([sys.executable, "app/main.py", "train", "--simulate", "--model-name", "TEMPLATE"])
    run([sys.executable, "app/main.py", "synth", "--simulate", "--model-name", "TEMPLATE"])
    run([sys.executable, "app/main.py", "eval", "--simulate", "--model-name", "TEMPLATE"])
