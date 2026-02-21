# Path: script/run_one.py
from fiberlen.pipeline import run_pipeline

if __name__ == "__main__":
    #result = run_pipeline("data/input/test3.tif", save_intermediate=True)
    result = run_pipeline("data/input/test1.png", save_intermediate=True)
    print(result)
