# Jetson X-ray Pneumonia Classifier

Jetson Nano (JetPack 4.6 / L4T R32.7.1) 위에서 X-ray 이미지 한 장을 업로드하면
TensorRT 엔진으로 NORMAL / PNEUMONIA 를 실시간 분류하는 데모입니다.

## 1. 프로젝트 개요
- 목적: 흉부 X-ray로 폐렴 여부를 Jetson 단에서 추론
- 데이터: `/nvdli-data/Pneumonia_Dataset/{NORMAL,PNEUMONIA}` 총 약 5,618장
- 모델: ResNet-18 (ImageNet pretrained → 2-class fine-tune)
- 가속: PyTorch → torch2trt → TensorRT FP16
- UI: Flask 웹 업로더 (파일 이름, 예측, 확률, inference time 표시)

## 2. 환경
- Jetson Nano / Xavier NX (JetPack 4.6, L4T r32.7.1)
- Docker image: `nvcr.io/nvidia/l4t-ml:r32.7.1-py3` (또는 커밋한 `jetson-xray:pneu`)
- Python 3.6
- Torch 1.10.0 (컨테이너 기본)
- TensorRT / pycuda (컨테이너 기본)
- 추가 Python: `numpy`, `pillow`, `matplotlib`, `scikit-learn`, `pyyaml`, `flask`, `opencv-python`, `torch2trt`

## 3. 실행 순서

```bash
# 0) 컨테이너 실행
sudo docker run --runtime nvidia -it --rm --network host \
  --shm-size=1g \
  -v /home/jongbum/nvdli-data:/nvdli-data \
  -v /home/jongbum/jetson-xray-pneumonia:/workspace \
  jetson-xray:pneu    # 또는 nvcr.io/nvidia/l4t-ml:r32.7.1-py3

cd /workspace

# 1) 데이터 split
python3 scripts/split_dataset.py --config configs/default.yaml

# 2) 학습
python3 scripts/train.py --config configs/default.yaml

# 3) 평가
python3 scripts/evaluate.py --config configs/default.yaml

# 4) TensorRT 엔진 생성
python3 scripts/convert_trt.py configs/default.yaml

# 5) 웹앱
PYTHONPATH=/workspace/scripts python3 app/app.py
# 브라우저: http://<jetson-ip>:5000

## 4. 설치 (Jetson 컨테이너 내부)
pip3 install -r requirements.txt
cd torch2trt && python3 setup.py install


## 5. 참고
- models/model_fp16.engine 은 Jetson에서 직접 생성해서 사용
- 이 프로젝트는 데모/교육용이며 실제 의료 진단용이 아님
