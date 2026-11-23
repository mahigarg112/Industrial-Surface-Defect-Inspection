"""Verify real tensor shapes and export a static architecture figure."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .common import OUT, SmallUNet, write_json


def main():
    torch.set_num_threads(2)
    model = SmallUNet().eval()
    result = {"parameters":sum(p.numel() for p in model.parameters()),"shapes":{}}
    for mode,h,w in [("whole",128,512),("patch",256,256)]:
        shapes,handles = {},[]
        for name in ["enc1","enc2","enc3","bottleneck","dec3","dec2","dec1","output"]:
            def hook(module,inputs,output,key=name):
                shapes[key] = list(output.shape)
            handles.append(getattr(model,name).register_forward_hook(hook))
        with torch.inference_mode():
            out = model(torch.zeros(1,3,h,w))
        assert list(out.shape) == [1,4,h,w]
        for handle in handles:
            handle.remove()
        result["shapes"][mode] = shapes
    write_json(OUT/"architecture_verified.json",result)
    fig,ax = plt.subplots(figsize=(12,7))
    ax.set(xlim=(0,12),ylim=(0,8))
    ax.axis("off")
    blocks = [(1,6.7,"Native image\n3 x 256 x 1600"),(4,6.7,"12 native patches\n3 x 256 x 256"),
              (4,5.2,"enc1\n16 x 256 x 256"),(4,3.7,"enc2\n32 x 128 x 128"),(4,2.2,"enc3\n64 x 64 x 64"),
              (6,0.7,"bottleneck\n128 x 32 x 32"),(8,2.2,"dec3\n64 x 64 x 64"),(8,3.7,"dec2\n32 x 128 x 128"),
              (8,5.2,"dec1\n16 x 256 x 256"),(10.5,5.2,"Output + sigmoid\n4 x 256 x 256"),
              (10.5,6.7,"Hann reconstruction\nthen 4 locked thresholds\n4 x 256 x 1600")]
    artists = []
    for x,y,label in blocks:
        artists.append(ax.text(x,y,label,ha="center",va="center",fontsize=9,bbox=dict(boxstyle="round,pad=0.45",facecolor="#dce9f4",edgecolor="#40647e")))
    edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(2,8),(3,7),(4,6)]
    for source,target in edges:
        x,y,_ = blocks[source]
        u,v,_ = blocks[target]
        skip = target-source in [6,4,2] and source in [2,3,4]
        ax.annotate("",xy=(u,v),xytext=(x,y),arrowprops=dict(arrowstyle="->",color="#ae7a2d" if skip else "#40647e",linestyle="--" if skip else "-",patchA=artists[source].get_bbox_patch(),patchB=artists[target].get_bbox_patch(),shrinkA=5,shrinkB=5))
    ax.set_title(f"Resolution-aware Small U-Net | {result['parameters']:,} parameters | shapes exclude batch",pad=15)
    fig.tight_layout()
    fig.savefig(OUT/"architecture.png",dpi=180)
    plt.close(fig)
    print(f"Verified architecture and exported diagram: {result['parameters']:,} parameters")


if __name__ == "__main__":
    main()
