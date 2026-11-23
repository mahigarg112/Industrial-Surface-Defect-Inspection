"""One two-epoch K continuation; only the auxiliary loss is new."""
from pathlib import Path
import csv
import ctypes
import json
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from phase2.common import ROOT,OUT,CLASSES,load_model,seed_all,sha,write_json
from phase2.data import TrainDataset,records
from phase2.augmentation_study.diagnostics import evaluate,PROTOCOL as EVAL_PROTOCOL
from .loss import loss_terms

HERE=Path(__file__).resolve().parent


def main():
    protocol=json.loads((HERE/'protocol.json').read_text())
    assert json.loads((HERE/'unit_tests.json').read_text())['passed']
    exclusive=json.loads((HERE/'training_annotation_exclusivity.json').read_text())
    assert exclusive['images']==9048 and exclusive['overlapping_GT_pixels']==0
    assert sha(HERE/'training_annotation_exclusivity.json')==protocol['exclusivity_sha256']
    assert EVAL_PROTOCOL['thresholds']==protocol['thresholds'] and EVAL_PROTOCOL['fixed_panel_ids']==protocol['fixed_panel_ids']
    assert sha(ROOT/'phase2/augmentation_study/diagnostics.py')==protocol['evaluation_source_sha256']
    for path,key in [(OUT/'final/model.pt','initial_checkpoint_sha256'),
                     (OUT/'K_extended_patch/last_training_state.pt','initial_state_sha256'),
                     (OUT/'partition.csv','partition_sha256')]:assert sha(path)==protocol[key]
    directory=HERE/protocol['candidate']
    if directory.exists():raise FileExistsError('Never overwrite a training run')
    directory.mkdir();seed_all(42)
    assert torch.cuda.is_available()
    model,checkpoint=load_model(OUT/'final/model.pt','cuda')
    state=torch.load(OUT/'K_extended_patch/last_training_state.pt',map_location='cpu',weights_only=True)
    assert state['epoch']==checkpoint['epoch']==9
    assert all(torch.equal(v.cpu(),state['model_state_dict'][k]) for k,v in model.state_dict().items())
    optimizer=torch.optim.Adam(model.parameters(),lr=.0001);optimizer.load_state_dict(state['optimizer'])
    for group in optimizer.param_groups:
        group['lr']=.0001
        assert group['betas']==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==0
    scaler=torch.amp.GradScaler('cuda');scaler.load_state_dict(state['scaler'])
    torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);del state
    weights=torch.tensor(checkpoint['config']['dice_class_weights'],device='cuda',dtype=torch.float32)
    source_paths=['phase2/overlap_regularization_study/train.py','phase2/overlap_regularization_study/loss.py',
                  'phase2/augmentation_study/diagnostics.py','phase2/data.py','phase2/objectives.py','phase2/patches.py']
    config={**checkpoint['config'],'epochs':11,'learning_rate':.0001,'continuation_parent':'K_extended_patch',
            'continuation_start_epoch':9,'continuation_state_sha256':protocol['initial_state_sha256'],
            'augmentation':None,'auxiliary_coefficient':protocol['coefficient'],'auxiliary_formula':protocol['auxiliary_formula'],
            'matched_reference':protocol['reference'],'max_additional_epochs':2,'attempted_update_budget':2262,
            'selection':protocol['selection'],'fixed_thresholds':protocol['thresholds'],
            'protocol_sha256':sha(HERE/'protocol.json'),'source_sha256':{p:sha(ROOT/p) for p in source_paths}}
    write_json(directory/'config.json',config)
    for path in source_paths:(directory/('source_'+Path(path).name)).write_bytes((ROOT/path).read_bytes())
    rows=records('train');history=[];best=-1.;useful=.23922878227237032;bad=attempted=skipped=0
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for epoch in [9,10]:
        dataset=TrainDataset(rows,'patch',epoch=epoch,foreground_crops=True)
        loader=DataLoader(dataset,batch_size=8,shuffle=True,num_workers=2,pin_memory=True,
                          generator=torch.Generator().manual_seed(42+epoch))
        model.train();sums=np.zeros(3,np.float64);seen=0;pixels=np.zeros(4,np.int64);pairs=np.zeros(4,np.int64)
        tick=time.perf_counter()
        for index,(images,masks) in enumerate(loader):
            assert not (masks.sum(1)>1).any(),'Training masks lost exclusivity'
            pixels+=masks.sum((0,2,3)).numpy().astype(np.int64)
            pairs+=masks.flatten(2).any(2).sum(0).numpy().astype(np.int64)
            images,masks=images.cuda(non_blocking=True),masks.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                total,baseline,auxiliary=loss_terms(model(images),masks,weights,protocol['coefficient'])
            if not torch.isfinite(total):raise RuntimeError('Nonfinite total loss')
            scaler.scale(total).backward();old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
            skipped+=int(scaler.get_scale()<old_scale);attempted+=1
            sums+=np.array([float(x.detach()) for x in [total,baseline,auxiliary]])*len(images);seen+=len(images)
            if (index+1)%100==0:print(f'P epoch{epoch+1} train {index+1}/{len(loader)} total={sums[0]/seen:.6f} auxiliary={sums[2]/seen:.6f}',flush=True)
        train_seconds=time.perf_counter()-tick
        exposure={'samples':seen,'positive_samples_by_class':pairs.tolist(),'positive_pixels_by_class':pixels.tolist()}
        ref=json.loads((ROOT/f'phase2/improvement_sprint/L_low_lr/epoch_{epoch+1}_target_exposure.json').read_text())
        assert exposure==ref,'Crop/source/target exposure differs from matched L'
        write_json(directory/f'epoch_{epoch+1}_target_exposure.json',exposure)
        evaluation=evaluate({'P':model},f'P_epoch{epoch+1}',reset_seed=False)
        metric=evaluation['models']['P']['metrics'];write_json(directory/f'epoch_{epoch+1}_validation.json',evaluation)
        row={'epoch':epoch+1,'train_loss':sums[0]/seen,'train_baseline_loss':sums[1]/seen,
             'train_auxiliary_raw':sums[2]/seen,'train_auxiliary_weighted':protocol['coefficient']*sums[2]/seen,
             'val_macro_dice':metric['macro_dice'],'val_macro_iou':metric['macro_iou'],
             'normal_false_alarm_rate':metric['normal_false_alarm_rate'],'learning_rate':.0001,
             'attempted_steps':attempted,'skipped_steps':skipped,'train_seconds':train_seconds,
             'validation_seconds':evaluation['seconds']}
        row.update({f'dice_class_{c}':metric['per_class'][str(c)]['dice'] for c in range(1,5)})
        history.append(row)
        with (directory/'history.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(history)
        if metric['macro_dice']>best:
            best=metric['macro_dice']
            torch.save({'model_state_dict':model.state_dict(),'config':config,'class_names':CLASSES,
                        'epoch':epoch+1,'val_macro_dice':best,'validation':metric,'smoke_test_only':False},directory/'best_model.pt')
            write_json(directory/'best_validation.json',evaluation)
        if metric['macro_dice']>=useful+.002:useful,bad=metric['macro_dice'],0
        else:bad+=1
        torch.save({'model_state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                    'epoch':epoch+1,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                    'config':config},directory/'last_training_state.pt')
        info={'status':'running','epochs_completed':len(history),'best_validation_macro_dice':best,
              'attempted_steps':attempted,'successful_steps':attempted-skipped,'skipped_steps':skipped,
              'training_seconds':sum(r['train_seconds'] for r in history),'duration_seconds':time.perf_counter()-started,
              'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
              'checkpoint_sha256':sha(directory/'best_model.pt')}
        write_json(directory/'run_info.json',info);print(json.dumps(row,indent=2),flush=True)
        if bad>=2:print('Same patience-two rule as L reached; two-epoch cap',flush=True);break
    info['status']='completed';write_json(directory/'run_info.json',info)


if __name__=='__main__':
    awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)) if os.name=='nt' else False
    try:main()
    finally:
        if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
