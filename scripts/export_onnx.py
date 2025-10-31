# export_onnx.py
import torch, argparse, yaml
from train import build_model
def main(cfg):
    ckpt = torch.load("models/best.ckpt", map_location="cpu")
    model = build_model(cfg['train']['model']); model.load_state_dict(ckpt["model"]); model.eval()
    dummy = torch.randn(1,3,cfg['data']['img_size'],cfg['data']['img_size'])
    onnx_path = cfg['export']['onnx_path']
    torch.onnx.export(model, dummy, onnx_path, input_names=["input"], output_names=["logits"], opset_version=cfg['export']['opset'], dynamic_axes={"input":{0:"N"}, "logits":{0:"N"}})
    print("exported:", onnx_path)
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/default.yaml"); args=ap.parse_args()
    cfg=yaml.safe_load(open(args.config)); main(cfg)

