import sys, os, traceback, time
sys.path.append("/workspace/scripts")

from flask import Flask, request, render_template_string
import yaml
import numpy as np

import pycuda.driver as cuda
import tensorrt as trt

# CUDA 1회 초기화
cuda.init()
dev = cuda.Device(0)

# 우리 TRT 추론 유틸
from infer_trt import load_image, infer

# 설정 로드
with open("configs/default.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

ENGINE_PATH = cfg["trt"]["engine_path"]
IMG_SIZE = cfg["data"]["img_size"]

TEMPLATE = """
<!doctype html>
<title>Jetson X-ray Pneumonia Demo</title>
<h2>Upload chest X-ray</h2>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
{% if error %}
  <h3 style="color:red;">Error: {{ error }}</h3>
{% endif %}
{% if filename %}
  <p>Uploaded file: <b>{{ filename }}</b></p>
{% endif %}
{% if pred is not none %}
  <h3>Prediction: {{ pred }}</h3>
  <p>Prob (NORMAL): {{ p0 }}</p>
  <p>Prob (PNEUMONIA): {{ p1 }}</p>
{% endif %}
{% if infer_ms is not none %}
  <p>Inference time: {{ infer_ms }} ms</p>
{% endif %}
"""

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    pred = None
    p0 = p1 = None
    error = None
    filename = None
    infer_ms = None

    if request.method == "POST":
        f = request.files.get("file")
        if not f:
            error = "no file uploaded"
            return render_template_string(TEMPLATE, pred=pred, p0=p0, p1=p1,
                                          error=error, filename=filename, infer_ms=infer_ms)

        filename = f.filename
        upload_path = "/tmp/xray_upload.png"
        f.save(upload_path)

        # 요청마다 CUDA 컨텍스트
        ctx = dev.make_context()
        try:
            x = load_image(upload_path, IMG_SIZE)

            t0 = time.time()
            out = infer(ENGINE_PATH, x)
            t1 = time.time()
            infer_ms = int((t1 - t0) * 1000)

            exp = np.exp(out - np.max(out))
            prob = exp / exp.sum()

            thr = 0.6
            if prob[1] >= thr:
                pred = "PNEUMONIA"
            else:
                pred = "NORMAL"

            p0 = f"{prob[0]:.4f}"
            p1 = f"{prob[1]:.4f}"

        except Exception as e:
            print("[app] inference error:", e)
            traceback.print_exc()
            error = str(e)
        finally:
            ctx.pop()

    return render_template_string(TEMPLATE, pred=pred, p0=p0, p1=p1,
                                  error=error, filename=filename, infer_ms=infer_ms)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=False)
