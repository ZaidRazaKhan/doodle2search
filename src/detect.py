# Python modules
import torch
from torch.autograd.variable import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import cv2


from models.models import *
from utils import *
from datasets import *
from torchvision import datasets
from yolo_detection import YoloDetector
from MSCocoDataloader import MSCocoDataloader
from PIL import Image

import os
import numpy as np
import time
from sklearn.metrics import average_precision_score
import math
import multiprocessing
from joblib import Parallel, delayed
import pickle

# Own modules
from scipy.spatial.distance import cdist
from options import Options
from utils import load_checkpoint, rec, precak
from models.encoder import EncoderCNN
from data.generator_train import load_data

def testcoco(im_loader, sketch, model, args, yolo_detector, transform):
    classes = load_classes('/DATA/khan.2/khan.2/doodle2search/src/coco.names')
    im_net,sk_net = model
    im_net.eval()
    sk_net.eval()
    torch.set_grad_enabled(False)

    ms_coco_data_loader = MSCocoDataloader('/DATA/dataset/mscoco/train2017/*')
    
    sk_feat, _ = sk_net(transform(sketch).cuda()[None, ...])
    acc_sk_em = sk_feat.cpu().data.numpy()
    similarity=np.zeros(len(im_loader))


    image_paths = []
    ap = []
    for batch_i, (img_paths, input_imgs) in enumerate(im_loader):
        
        if batch_i%1000==0:
            print(batch_i)
        image_paths.append(img_paths)
        #print(np.array(input_imgs).shape)
        if np.array(input_imgs).shape[1]!=3:
            similarity[batch_i] = 0
            continue
        detections = yolo_detector.get_detections(input_imgs)
        if detections is None:
            similarity[batch_i] = 0
            continue
        img = ms_coco_data_loader[batch_i]
        acc_im_em = np.array([])
        for detection in detections:
            if detection == None:
                continue
            detection = rescale_boxes(detection, 416, (np.array(img).shape)[:2])
            for k in range(len(detection)):
                x1, y1, x2, y2, conf, cls_conf, cls_pred = detection[k].numpy()
                detected_class = classes[int(cls_pred)]
                if detected_class == 'motorbike':
                    ap.append(1)
                subim = img.crop((x1, y1, x2, y2))
                subim = Image.fromarray(cv2.resize(np.array(subim.convert('RGB')), (224, 224)))
                #print(np.array(subim).shape)
                subim = transform(subim).cuda()
            
                subim_feat, _ = im_net(subim[None, ...])
                if k == 0:
                    acc_im_em = subim_feat.cpu().data.numpy()
                else:
                    acc_im_em = np.concatenate((acc_im_em, subim_feat.cpu().data.numpy()), axis=0)

        distance = cdist(acc_sk_em, acc_im_em, 'euclidean')
        sim = 1/(1+distance)
        similarity[batch_i]=np.max(sim)
        if len(ap) != batch_i:
            ap.append(0)
    index_of_image = (-similarity).argsort()
    image_retrieved = []
    average_precesion = []
    result = []
    for i in range(10):
        image_retrieved.append(image_paths[(index_of_image[i])])
        result.append(similarity[index_of_image[i]])
        average_precesion.append(ap[index_of_image[i]])
        print(image_retrieved[i])
    average_precesion = np.array(average_precesion)
    result = np.array(result)
    print(average_precesion)
    print(result)
    print(average_precision_score(average_precesion.reshape((-1)), result.reshape((-1))))
    

def main():
    weights_path = '/DATA/khan.2/khan.2/doodle2search/src/yolov3.weights'
    model_def = '/DATA/khan.2/khan.2/doodle2search/src/yolov3.cfg'
    img_size = 416
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_model = Darknet(model_def, img_size=img_size).to(device)
    if weights_path.endswith(".weights"):
        # Load darknet weights
        yolo_model.load_darknet_weights(weights_path)
    else:
        # Load checkpoint weights
        yolo_model.load_state_dict(torch.load(weights_path))

    yolo_model.eval()
    

    # ms_coco_data_loader = DataLoader(MSCocoDataloader('/DATA/dataset/mscoco/train2017/*'), batch_size = 1, shuffle = False, num_workers = 0)
    yolo_detector = YoloDetector(yolo_model)
    ms_coco_data_loader = DataLoader(
        ImageFolder('/DATA/dataset/mscoco/train2017', img_size=416),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )


    print('Prepare data')
    transform = transforms.Compose([transforms.ToTensor()])
    sketch = Image.fromarray(cv2.resize(np.array(Image.open('/DATA/khan.2/khan.2/doodle2search/n03790512_12141-1.png').convert('RGB')), (224, 224)))
    # print(type(sketch))
    # sketch = Image.open('/DATA/khan.2/khan.2/doodle2search/src/n07739125_10025-1.png')
    #print(sketch)


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

    print('***Test***')
    testcoco(ms_coco_data_loader, sketch, [im_net, sk_net], args, yolo_detector, transform)


if __name__ == '__main__':
    # Parse options
    args = Options(test=True).parse()
    print('Parameters:\t' + str(args))

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

