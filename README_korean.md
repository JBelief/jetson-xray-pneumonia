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

## 3. 데이터셋 준비

본 프로젝트에서 사용한 흉부 X-ray 데이터셋은 Kaggle의 공개 데이터셋을 사용했습니다.

- 출처: https://www.kaggle.com/datasets/sangeethakallat/chest-radiograph-images-pneumonia-and-normal
- 구성: Pneumonia / Normal X-ray 이미지
- 라이선스/사용 조건은 Kaggle 원본 페이지를 따릅니다.

Jetson에서 다운로드할 경우 예시:

```bash
curl -L -o chest-radiograph-images-pneumonia-and-normal.zip \
  https://www.kaggle.com/api/v1/datasets/download/sangeethakallat/chest-radiograph-images-pneumonia-and-normal

unzip chest-radiograph-images-pneumonia-and-normal.zip -d /home/jongbum/nvdli-data/Pneumonia_Dataset


이후 configs/default.yaml 에서 다음 경로를 사용하도록 했습니다.
data:
  pneumonia_dir: "/nvdli-data/Pneumonia_Dataset/PNEUMONIA"
  normal_dir: "/nvdli-data/Pneumonia_Dataset/NORMAL"


## 4. 실행 순서

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

## 5. 설치 (Jetson 컨테이너 내부)
pip3 install -r requirements.txt
cd torch2trt && python3 setup.py install


## 6. 참고
- models/model_fp16.engine 은 Jetson에서 직접 생성해서 사용
- 이 프로젝트는 데모/교육용이며 실제 의료 진단용이 아님
