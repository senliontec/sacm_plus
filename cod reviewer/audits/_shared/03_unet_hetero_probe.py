import sys, json, torch, torch.nn as nn
from pathlib import Path
ROOT=Path('/home/dev/mywork/project/segment'); sys.path.insert(0,str(ROOT/'sacm_plus/src'))
from baseline_zoo.models.unet import UNet as PortUNet
from baseline_zoo.adapters.unet import dice_loss as port_dice
sys.path.insert(0,str(ROOT/'3rd/Pytorch-UNet'))
from unet.unet_model import UNet as OffUNet
from utils.dice_score import dice_coeff as off_dice_coeff, dice_loss as off_dice_loss
torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
torch.manual_seed(11)
off=OffUNet(3,1); port=PortUNet(3,1); port.load_state_dict(off.state_dict()); off.eval(); port.eval()
g=torch.Generator().manual_seed(7)
x=torch.randn(4,3,64,64,generator=g)
t=torch.zeros(4,1,64,64)
t[1]=1.0
t[2,0,::4,::3]=1.0
t[3,0,5:45,5:45]=1.0
with torch.no_grad(): yo=off(x); yp=port(x)
prob_o=torch.sigmoid(yo.squeeze(1)); prob_p=torch.sigmoid(yp.squeeze(1))
# official per-image (dim=2 -> spatial only) and global (reduce_batch_first=True)
per=[float(off_dice_coeff(prob_o[i],t[i,0],reduce_batch_first=False)) for i in range(4)]
glob=float(off_dice_coeff(prob_o,t.squeeze(1),reduce_batch_first=True))
portmean=1.0-float(port_dice(yp,t))
# port's per-image vector
inter=2*(prob_p*t.squeeze(1)).sum((1,2)); sets=prob_p.sum((1,2))+t.squeeze(1).sum((1,2)); sets=torch.where(sets==0,inter,sets); perport=((inter+1e-6)/(sets+1e-6)).tolist()
crit=nn.BCEWithLogitsLoss()
yo2=off(x); yp2=port(x)
off.zero_grad(); lo=crit(yo2,t)+off_dice_loss(torch.sigmoid(yo2.squeeze(1)),t.squeeze(1)); lo.backward()
port.zero_grad(); lp=crit(yp2,t)+port_dice(yp2,t); lp.backward()
gd=max(float((a.grad-b.grad).abs().max()) for a,b in zip(off.parameters(),port.parameters()) if a.grad is not None and b.grad is not None)
out={'per_image_official':per,'per_image_port':perport,'official_global_dice':glob,'port_mean_dice':portmean,
     'dice_diff':abs(glob-portmean),'loss_official':float(lo),'loss_port':float(lp),'loss_absdiff':abs(float(lo)-float(lp)),
     'grad_maxdiff':gd,'forward_maxdiff':float((yo-yp).abs().max())}
out2={}
for r in range(2):
    yo3=off(x); yp3=port(x)
    off.zero_grad(); lo3=crit(yo3,t)+off_dice_loss(torch.sigmoid(yo3.squeeze(1)),t.squeeze(1)); lo3.backward()
    port.zero_grad(); lp3=crit(yp3,t)+port_dice(yp3,t); lp3.backward()
    gd3=max(float((a.grad-b.grad).abs().max()) for a,b in zip(off.parameters(),port.parameters()) if a.grad is not None and b.grad is not None)
    out2[r]={'loss_official':float(lo3),'loss_port':float(lp3),'grad_maxdiff':gd3}
out['repeat_runs']=out2
print(json.dumps(out,indent=2))
