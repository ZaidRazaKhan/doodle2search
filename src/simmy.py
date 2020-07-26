# Python modules
import torch
from torch.autograd.variable import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from PIL import Image
import os
import numpy as np
import time
from sklearn.metrics import average_precision_score
import math
import multiprocessing
from joblib import Parallel, delayed
import pickle
import cv2
# Own modules
from scipy.spatial.distance import cdist
from options import Options
from utils import load_checkpoint, rec, precak
from models.encoder import EncoderCNN
from data.generator_train import load_data
def testcoco(image, sketch, model, args, transform):
    im_net,sk_net = model
    im_net.eval()
    sk_net.eval()
    torch.set_grad_enabled(False)

    sk_feat, _ = sk_net(transform(sketch).cuda()[None, ...])
    acc_sk_em = sk_feat.cpu().data.numpy()
    im_feat, _ = im_net(transform(image).cuda()[None, ...])
    acc_im_em = im_feat.cpu().data.numpy()
        
    distance = cdist(acc_sk_em, acc_im_em, 'euclidean')
    sim = 1/(1+distance)
    print(sim)


def main():
    print('Preparing data')
    transform = transforms.Compose([transforms.ToTensor()])
    sketch = Image.fromarray(cv2.resize(np.array(Image.open('/DATA/khan.2/khan.2/doodle2search/n07739125_11148-1.png').convert('RGB')), (224, 224)))
    image = Image.fromarray(cv2.resize(np.array(Image.open('/DATA/khan.2/khan.2/doodle2search/n07739125_11148.jpg').convert('RGB')), (224, 224)))
    print('Create model')
    im_net = EncoderCNN(out_size=args.emb_size, attention=args.attn)
    sk_net = EncoderCNN(out_size=args.emb_size, attention=args.attn)


    if args.cuda:
        print('\t* CUDA')
        im_net, sk_net = im_net.cuda(), sk_net.cuda()

    print('Loading model')
    checkpoint = load_checkpoint(args.load)
    im_net.load_state_dict(checkpoint['im_state'])
    sk_net.load_state_dict(checkpoint['sk_state'])
    print('Loaded model at epoch {epoch} and mAP {mean_ap}%'.format(epoch=checkpoint['epoch'],
                                                                    mean_ap=checkpoint['best_map']))

    testcoco(image, sketch, [im_net, sk_net], args, transform)
    

if __name__ == '__main__':
    # Parse options
    args = Options(test=True).parse()
    # print('Parameters:\t' + str(args))

    # Check cuda & Set random seed
    args.cuda = args.ngpu > 0 and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Check Test and Load
    if args.load is None:
        raise Exception('Cannot test without loading a model.')

    main()

