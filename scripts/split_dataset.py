import argparse, os, random, yaml, glob
from pathlib import Path

def main(cfg):
    pne = sorted(glob.glob(os.path.join(cfg['data']['pneumonia_dir'], "*")))
    nor = sorted(glob.glob(os.path.join(cfg['data']['normal_dir'], "*")))
    data = [(p,1) for p in pne] + [(n,0) for n in nor]
    random.seed(42); random.shuffle(data)

    N = len(data)
    n_test = int(N * cfg['data']['test_ratio'])
    n_val  = int(N * cfg['data']['val_ratio'])
    test   = data[:n_test]
    val    = data[n_test:n_test+n_val]
    train  = data[n_test+n_val:]

    out = Path(cfg['data']['output_splits']); out.mkdir(parents=True, exist_ok=True)
    for name, subset in [('train',train),('val',val),('test',test)]:
        with open(out/f"{name}.txt","w") as f:
            for p,l in subset: f.write(f"{p}\t{l}\n")
    print(f"train:{len(train)} val:{len(val)} test:{len(test)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg)

