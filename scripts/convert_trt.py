# scripts/convert_trt.py
import torch, yaml
from train import build_model
from torch2trt import torch2trt

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 학습된 PyTorch 모델 로드
    ckpt = torch.load('models/best.ckpt', map_location='cpu')
    model = build_model(cfg['train']['model'])
    model.load_state_dict(ckpt['model'])
    model.eval().to(device)

    # 더미 입력 (이미지 사이즈 맞추기)
    H = W = cfg['data']['img_size']
    x = torch.randn(1, 3, H, W).to(device)

    # TensorRT 변환 (FP16)
    print('Converting to TensorRT (fp16=True)...')
    model_trt = torch2trt(
        model, [x],
        fp16_mode=True,
        max_workspace_size=1<<28  # 256MB, 필요시 키우기
    )

    # 엔진 직렬화 저장
    engine_path = cfg['trt']['engine_path']
    with open(engine_path, 'wb') as f:
        f.write(model_trt.engine.serialize())
    print('Saved TensorRT engine to:', engine_path)

if __name__ == '__main__':
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/default.yaml'
    main(cfg_path)

