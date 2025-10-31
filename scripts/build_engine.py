i# build_engine.py (TensorRT Python API)
import argparse, yaml, tensorrt as trt, os
def build_engine(onnx_path, engine_path, fp16=True, workspace=1<<30):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, 'rb') as f: parser.parse(f.read())
    config = builder.create_builder_config(); config.max_workspace_size = workspace
    if fp16 and builder.platform_has_fast_fp16: config.set_flag(trt.BuilderFlag.FP16)
    engine = builder.build_engine(network, config)
    with open(engine_path,"wb") as f: f.write(engine.serialize())
    print("saved engine:", engine_path)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/default.yaml"); args=ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    build_engine(cfg['export']['onnx_path'], cfg['trt']['engine_path'], fp16=cfg['trt']['fp16'])

