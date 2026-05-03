import os
import runpy
import sys

ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT, "src"))
runpy.run_path(os.path.join(ROOT, "scripts", "synthesize.py"), run_name="__main__")
