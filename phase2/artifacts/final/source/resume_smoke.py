"""Check a real-GPU two-update continuation against save/reload between updates."""
import argparse
import io
import json
import copy

import torch

from .common import OUT,SmallUNet,seed_all,sha,write_json
from .data import records,TrainDataset
from .objectives import present_dice_loss,balanced_present_dice_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent",required=True)
    args = parser.parse_args()
    parent = OUT/args.parent
    info = json.loads((parent/"run_info.json").read_text())
    assert info["status"] == "completed"
    state_path = parent/"last_training_state.pt"
    original_digest = sha(state_path)
    state = torch.load(state_path,map_location="cpu",weights_only=True)
    config = state["config"]
    assert config["mode"] == "patch" and ("GT-positive" in config["loss"] or "positive-pair Dice" in config["loss"])
    seed_all(config["seed"])
    rows = records("train")
    chosen = [next(row for row in rows if [i for i,rle in enumerate(row["rle"].split("|")) if rle] == [c]) for c in range(4)]
    chosen += [row for row in rows if not any(row["rle"].split("|"))][:4]
    pairs = [TrainDataset(chosen,"patch",epoch=state["epoch"],foreground_crops=True)[i] for i in range(8)]
    images = torch.stack([p[0] for p in pairs]).cuda()
    targets = torch.stack([p[1] for p in pairs]).cuda()
    weights = torch.tensor(config["dice_class_weights"],device="cuda") if "dice_class_weights" in config else None

    def restore(value):
        model = SmallUNet().cuda().train()
        model.load_state_dict(value["model_state_dict"],strict=True)
        optimizer = torch.optim.Adam(model.parameters(),lr=config["learning_rate"])
        # CPU Adam step tensors can share storage with the supplied dictionary.
        # Each independent comparison path needs its own source optimizer state.
        optimizer.load_state_dict(copy.deepcopy(value["optimizer"]))
        scaler = torch.amp.GradScaler("cuda")
        scaler.load_state_dict(value["scaler"])
        torch.set_rng_state(value["torch_rng"])
        torch.cuda.set_rng_state_all(value["cuda_rng"])
        return model,optimizer,scaler

    def update(model,optimizer,scaler):
        for _ in range(8):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16):
                logits = model(images)
                loss = present_dice_loss(logits,targets) if weights is None else balanced_present_dice_loss(logits,targets,weights)
            assert torch.isfinite(loss)
            scaler.scale(loss).backward()
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale:
                return float(loss.detach())
        raise AssertionError("No finite continuation update")

    uninterrupted = restore(state)
    losses_a = [update(*uninterrupted),update(*uninterrupted)]
    split = restore(state)
    losses_b = [update(*split)]
    buffer = io.BytesIO()
    torch.save({"model_state_dict":split[0].state_dict(),"optimizer":split[1].state_dict(),
                "scaler":split[2].state_dict(),"torch_rng":torch.get_rng_state(),
                "cuda_rng":torch.cuda.get_rng_state_all()},buffer)
    buffer.seek(0)
    restored = restore(torch.load(buffer,map_location="cpu",weights_only=True))
    losses_b.append(update(*restored))
    assert losses_a == losses_b
    assert all(torch.equal(v,restored[0].state_dict()[k]) for k,v in uninterrupted[0].state_dict().items())
    assert any(not torch.equal(v.cpu(),state["model_state_dict"][k]) for k,v in uninterrupted[0].state_dict().items())
    first,second = uninterrupted[1].state_dict(),restored[1].state_dict()
    assert first["param_groups"] == second["param_groups"]
    for key,values in first["state"].items():
        for name,value in values.items():
            assert torch.equal(value,second["state"][key][name]) if torch.is_tensor(value) else value == second["state"][key][name]
    assert uninterrupted[2].state_dict() == restored[2].state_dict()
    assert sha(state_path) == original_digest
    result = {"passed":True,"parent":parent.name,"parent_state_sha256":original_digest,
              "parent_config_sha256":sha(parent/"config.json"),"partition_sha256":sha(OUT/"partition.csv"),
              "two_successful_updates_each_path":True,"losses_identical":True,
              "actual_two_update_losses":losses_a,
              "weights_adam_and_scaler_identical":True,"parent_unchanged":True,
              "scope":"mechanical resume equivalence on real training crops; not an accuracy experiment"}
    write_json(OUT/"resume_preflight.json",result)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    main()
