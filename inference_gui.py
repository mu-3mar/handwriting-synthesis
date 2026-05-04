import os
import runpy
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
runpy.run_path(os.path.join(ROOT, "scripts", "inference_gui.py"), run_name="__main__")
