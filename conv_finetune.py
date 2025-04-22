import os, argparse, copy, time, torch, h5py
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from nn_models.custom_conv import CustomConv1D
from nn_models.utils import train_utils       
from nn_models.utils import dloader_utils     

def load_conv_model(model_spec, ckpt_path, freeze='none', k=1):
    model = CustomConv1D(**model_spec)
    sd = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(sd['model'])                 
    if freeze == 'head':
        for p in model.parameters():
            p.requires_grad = False
        for p in model.lin_out.parameters():
            p.requires_grad = True
    elif freeze == 'firstK':
        for name, p in model.named_parameters():
            if name.startswith('conv') and int(name[4]) < k:
                p.requires_grad = False
    # freeze == 'none' → do nothing
    return model
# ------------------------------------------------------------------------


def arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--data',   required=True, help='HDF5 file with training data')
    p.add_argument('--ckpt',   required=True, help='pre‑trained model checkpoint')
    p.add_argument('--freeze', default='none', choices=['none', 'head', 'firstK'])
    p.add_argument('--k',      type=int, default=1, help='K conv layers to freeze (firstK)')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch',  type=int, default=16)
    p.add_argument('--lr',     type=float, default=3e-4)
    p.add_argument('--gpu',    action='store_true')
    return p.parse_args()


def build_dataset(h5_path, seq_len=200):
    """Very simple loader: concatenates all subjects, returns TensorDataset"""
    xs, ys = [], []
    with h5py.File(h5_path, 'r') as f:
        for subj in f.keys():
            x_pel = f[f'{subj}/pelvis/acc'][:]
            x_lth = f[f'{subj}/lthigh/acc'][:]
            y_ang = f[f'{subj}/lknee/angle'][:]
            n_chunks = x_pel.shape[0] // seq_len
            if n_chunks == 0: continue
            x = np.concatenate([x_pel, x_lth], axis=1)[:n_chunks*seq_len]
            y = y_ang[:n_chunks*seq_len]
            xs.append(x.reshape(n_chunks, seq_len, -1))
            ys.append(y.reshape(n_chunks, seq_len, -1))
    x_all = torch.from_numpy(np.vstack(xs)).float()
    y_all = torch.from_numpy(np.vstack(ys)).float()
    return TensorDataset(x_all, y_all)


def main():
    args = arg_parser()

    model_spec = {'inp_size':[16], 'outp_size':[3], 'window':41,
                  'conv_dropout':[0,0,0], 'conv_batchnorm':True,
                  'lin_activation':['Sigmoid'], 'prediction':'angle'}
    
    model = load_conv_model(model_spec, args.ckpt,
                            freeze=args.freeze, k=args.k)
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available()
                          else 'cpu')
    model.to(device)

    trainable = filter(lambda p: p.requires_grad, model.parameters())
    optim = torch.optim.Adam(trainable, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.98)
    criterion = torch.nn.MSELoss()

    ds = build_dataset(args.data)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True)

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
        scheduler.step()
        print(f'Epoch {epoch+1}/{args.epochs} – loss {epoch_loss/len(loader):.4f}')

    # save new checkpoint
    stamp = time.strftime('%Y%m%d_%H%M%S')
    save_path = f'finetuned_conv_{stamp}.pt'
    torch.save({'model': model.state_dict()}, save_path)
    print('finished – weights in', save_path)


if __name__ == '__main__':
    main()

