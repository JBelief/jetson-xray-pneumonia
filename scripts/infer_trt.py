import sys, os, yaml
import numpy as np
from PIL import Image

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

def load_image(path, size):
    # 1) 열고 리사이즈
    img = Image.open(path).convert("RGB").resize((size, size))

    # 2) 0~1 로 스케일 (H, W, C)
    arr = np.array(img, dtype=np.float32) / 255.0

    # 3) CHW
    arr = arr.transpose(2, 0, 1)   # (3, H, W)

    # 4) 배치 차원 추가 → (1, 3, H, W)
    arr = np.expand_dims(arr, 0)

    # 5) ImageNet 정규화
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    arr = (arr - mean) / std

    # 6) TensorRT로 보낼 때는 반드시 contiguous
    arr = np.ascontiguousarray(arr)

    return arr

def infer(engine_path, inp):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # I/O 버퍼 준비
    d_inputs = []
    d_outputs = []
    host_outputs = []

    stream = cuda.Stream()

    for binding in engine:
        idx = engine.get_binding_index(binding)
        shape = context.get_binding_shape(idx)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        size = int(np.prod(shape))
        if engine.binding_is_input(binding):
            d_in = cuda.mem_alloc(inp.nbytes)
            d_inputs.append(d_in)
        else:
            host_out = np.empty(size, dtype=dtype)
            d_out = cuda.mem_alloc(host_out.nbytes)
            d_outputs.append(d_out)
            host_outputs.append(host_out)

    # 복사 → 실행 → 복사
    cuda.memcpy_htod_async(d_inputs[0], inp, stream)
    context.execute_async_v2(bindings=[int(d_inputs[0])] + [int(d_outputs[0])],
                             stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_outputs[0], d_outputs[0], stream)
    stream.synchronize()

    return host_outputs[0]

if __name__ == "__main__":
    # 인자 처리
    if len(sys.argv) < 3:
        print("usage: python3 scripts/infer_trt.py --config configs/default.yaml --image /path/to/img")
        sys.exit(1)

    # 단순 파서
    cfg_path = None
    img_path = None
    for i, a in enumerate(sys.argv):
        if a == "--config":
            cfg_path = sys.argv[i+1]
        if a == "--image":
            img_path = sys.argv[i+1]

    if cfg_path is None or img_path is None:
        print("need --config and --image")
        sys.exit(1)

    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    img_size = cfg["data"]["img_size"]
    engine_path = cfg["trt"]["engine_path"]
    labels = cfg.get("app", {}).get("labels", ["NORMAL", "PNEUMONIA"])

    x = load_image(img_path, img_size)
    out = infer(engine_path, x)          # shape: (2,)
    # softmax
    exp = np.exp(out - np.max(out))
    prob = exp / exp.sum()

    pred_idx = int(prob.argmax())
    pred_label = labels[pred_idx] if pred_idx < len(labels) else str(pred_idx)
    print(f"image: {img_path}")
    print(f"pred : {pred_label}")
    print(f"probs: NORMAL={prob[0]:.4f}, PNEUMONIA={prob[1]:.4f}")
