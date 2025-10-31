# benchmark_fps.py
import time, yaml, numpy as np
from infer_trt import load_engine, preprocess, infer
import cv2, argparse

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/default.yaml"); ap.add_argument("--image"); ap.add_argument("--iters",type=int,default=200); args=ap.parse_args()
    cfg=yaml.safe_load(open(args.config))
    engine = load_engine(cfg['trt']['engine_path'])
    img = cv2.imread(args.image); x = preprocess(img, cfg['data']['img_size'])
    # warmup
    for _ in range(20): infer(engine, x)
    t0=time.time()
    for _ in range(args.iters): infer(engine, x)
    t1=time.time()
    fps = args.iters/(t1-t0)
    print(f"FPS: {fps:.2f}")

