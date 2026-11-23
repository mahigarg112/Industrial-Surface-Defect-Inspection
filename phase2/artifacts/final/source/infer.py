"""Run a locked selected model on a new steel image without ground truth."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .common import load_model, sha, write_json, seed_all
from .patches import predict


def run(package,image_path,output):
    package,image_path,output = Path(package),Path(image_path),Path(output)
    if output.exists():
        raise FileExistsError("Choose a fresh inference output directory")
    lock = json.loads((package/"lock.json").read_text())
    checkpoint = package/"model.pt"
    if sha(checkpoint) != lock["checkpoint_sha256"]:
        raise ValueError("Checkpoint does not match selection lock")
    model,ckpt = load_model(checkpoint)
    seed_all(ckpt["config"].get("seed",42))
    with Image.open(image_path) as f:
        image = np.asarray(f.convert("RGB")).copy()
    probability = predict(model,image,ckpt["config"]["mode"],stride=lock["stride"],kind=lock["blend"])
    prediction = probability >= np.asarray(lock["thresholds"])[:,None,None]
    if probability.shape != (4,*image.shape[:2]) or not np.isfinite(probability).all():
        raise RuntimeError("Invalid native-coordinate prediction")
    output.mkdir(parents=True)
    np.savez_compressed(output/"prediction.npz",probabilities=probability,masks=prediction)
    for c in range(4):
        Image.fromarray(prediction[c].astype(np.uint8)*255).save(output/f"class_{c+1}.png")
    colors = np.array([[255,60,60],[50,255,100],[60,150,255],[255,220,50]],np.float32)
    color = np.einsum("chw,ck->hwk",prediction.astype(np.float32),colors)
    active = prediction.sum(0)
    color /= np.maximum(active[:,:,None],1)
    overlay = np.where(active[:,:,None]>0,0.6*image+0.4*color,image).astype(np.uint8)
    Image.fromarray(overlay).save(output/"overlay.png")
    write_json(output/"inference.json",{"image":str(image_path),"image_sha256":sha(image_path),
               "native_probability_shape":list(probability.shape),"thresholds":lock["thresholds"],
               "checkpoint_sha256":sha(checkpoint),"positive_pixels_by_class":prediction.sum((1,2)).tolist(),
               "ground_truth_available":False,"note":"Successful inference demonstrates execution, not accuracy on this unlabeled image."})
    print(f"Saved native-resolution predictions: {output}",flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--package",type=Path,required=True)
    p.add_argument("--image",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a = p.parse_args()
    run(a.package,a.image,a.output)
