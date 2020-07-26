# Python modules
import torch
from torch.autograd.variable import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import cv2

from pycocotools.coco import COCO  
from models.models import *
from utils import *
from datasets import *
from torchvision import datasets
from yolo_detection import YoloDetector
from MSCocoDataloader import MSCocoDataloader
from PIL import Image

import os
import os.path
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
    im_net,sk_net = model
    im_net.eval()
    sk_net.eval()
    torch.set_grad_enabled(False)
    iop=0
    ms_coco_data_loader = MSCocoDataloader('/DATA/dataset/mscoco/train2017/*')
    # use of pycoco
    coco = COCO('src/instances_train2017.json')
    cat_name='cat'
    sk_feat, _ = sk_net(transform(sketch).cuda()[None, ...])
    acc_sk_em = sk_feat.cpu().data.numpy()
    similarity=np.zeros(len(im_loader))

    ground_truth=np.zeros(len(im_loader))
    image_paths = []
    for batch_i, (img_paths, input_imgs) in enumerate(im_loader):
        #if batch_i==1000:
        #    break

        if batch_i == 5000:
            print(batch_i)
        image_paths.append(img_paths)
        # code snipppet to get ground truth
        img_id=int(img_paths[0][-16:-4])
        annotation_ids = coco.getAnnIds(img_id)
        annotations = coco.loadAnns(annotation_ids)
        for i in range(len(annotations)):
            entity_id = annotations[i]["category_id"]
            entity = coco.loadCats(entity_id)[0]["name"]
            if entity==cat_name:
                ground_truth[batch_i]=1
                print(img_id)
                break
        #print(np.array(input_imgs).shape)
        if np.array(input_imgs).shape[1]!=3:
            similarity[batch_i] = 0
            continue
        detections = yolo_detector.get_detections(input_imgs)[0]
        if detections is None:
            similarity[batch_i] = 0
            continue
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join('../../../dataset/mscoco/train2017', path)).convert('RGB')
        #print(img.size)
        detections = rescale_boxes(detections, 416, (np.array(img).shape)[:2])
        acc_im_em = np.array([])
        for k in range(len(detections)):
            x1, y1, x2, y2, conf, cls_conf, cls_pred = detections[k].numpy()
            
            subim = img.crop((x1, y1, x2, y2))

            subim = Image.fromarray(cv2.resize(np.array(subim.convert('RGB')), (224, 224)))
            #if ground_truth[batch_i]:
            #    print(cls_pred)
            #    if cls_pred==49:
            #        iop=iop+1
            #        subim.save("apple"+str(iop)+".png")     
            #print(np.array(subim).shape)
            subim = transform(subim).cuda()
        
            subim_feat, _ = im_net(subim[None, ...])
            if k == 0:
                acc_im_em = subim_feat.cpu().data.numpy()
            else:
                acc_im_em = np.concatenate((acc_im_em, subim_feat.cpu().data.numpy()), axis=0)
            #print(batch_i,"srijan")    
        # print(acc_sk_em.shape,acc_im_em.shape)

        distance = cdist(acc_sk_em, acc_im_em, 'euclidean')
        #print(distance.shape)
        sim = 1/(1+distance)
        similarity[batch_i]=np.max(sim)
        if ground_truth[batch_i]:
            print(similarity[batch_i])
    
    index_of_image = (-similarity).argsort()
    image_retrieved = []
    for i in range(10):
        image_retrieved.append(image_paths[(index_of_image[i])])
        print(image_retrieved[i], similarity[index_of_image[i]])
    ap = average_precision_score(ground_truth,similarity)
    print(ap)
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
    classes = load_classes('/DATA/khan.2/khan.2/doodle2search/src/coco.names')

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
    # Cat
    # sketch = Image.fromarray(cv2.resize(np.array(Image.open('/DATA/khan.2/khan.2/doodle2search/n02121620_19341-1.png').convert('RGB')), (224, 224)))
    sketch = Image.fromarray(cv2.resize(np.array(Image.open('/DATA/khan.2/khan.2/doodle2search/n02121620_11995-1.png').convert('RGB')), (224, 224)))
    # Car
    # sketch = Image.fromarray(cv2.resize(np.array(Image.open('/DATA/khan.2/khan.2/doodle2search/n02958343_3561-1.png').convert('RGB')), (224, 224)))
    # sketch = Image.fromarray(cv2.resize(np.array(Image.open('/DATA/khan.2/khan.2/doodle2search/n02958343_10092-1.png').convert('RGB')), (224, 224)))
    # sketch = Image.fromarray(cv2.resize(np.array(Image.open('/DATA/khan.2/khan.2/doodle2search/n02958343_12252-1.png').convert('RGB')), (224, 224)))
    # Airplane
    # sketch = Image.fromarray(cv2.resize(np.array(Image.open('/DATA/khan.2/khan.2/doodle2search/n02691156_1142-2.png').convert('RGB')), (224, 224)))
    # sketch = Image.fromarray(cv2.resize(np.array(Image.open('/DATA/khan.2/khan.2/doodle2search/n02691156_2289-1.png').convert('RGB')), (224, 224)))
    # sketch = Image.fromarray(cv2.resize(np.array(Image.open('/DATA/khan.2/khan.2/doodle2search/n02691156_11016-1.png').convert('RGB')), (224, 224)))
    # zebra
    # sketch = Image.fromarray(cv2.resize(np.array(Image.open('/DATA/khan.2/khan.2/doodle2search/n02391049_11162-1.png').convert('RGB')), (224, 224)))
    # sketch = Image.fromarray(cv2.resize(np.array(Image.open('/DATA/khan.2/khan.2/doodle2search/n02391049_2136-1.png').convert('RGB')), (224, 224)))

    # sketch = Image.fromarray(cv2.resize(np.array(Image.open('/DATA/khan.2/khan.2/doodle2search/n03790512_12141-1.png').convert('RGB')), (224, 224)))
    # sketch = Image.fromarray(cv2.resize(np.array(Image.open('/DATA/khan.2/khan.2/doodle2search/n07739125_11148-1.png').convert('RGB')), (224, 224)))
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
