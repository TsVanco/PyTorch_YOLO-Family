from copy import deepcopy
from thop import profile
import argparse
import os

import torch

from config.yolo_config import yolo_config
from data.transforms import ValTransforms
from models.yolo import build_model
from utils.misc import TestTimeAugmentation

from evaluator.vocapi_evaluator import VOCAPIEvaluator
from evaluator.cocoapi_evaluator import COCOAPIEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    # basic
    parser.add_argument('-size', '--img_size', default=608, type=int,
                        help='img_size')
    parser.add_argument('--cuda', action='store_false', default=True,
                        help='Use cuda')
    # model
    parser.add_argument('-m', '--model', default='yolov3',
                        help='yolov1, yolov2, yolov3, yolov3_spp, yolov3_de, '
                        'yolov4, yolo_tiny, yolo_nano')
    parser.add_argument('--weight', type=str,
                        default='/home/tsvanco/OD/ckpts/yolov3_36.9_59.0.pth',
                        help='Trained state_dict file path to open')
    parser.add_argument('--conf_thresh', default=0.001, type=float,
                        help='NMS threshold')
    parser.add_argument('--nms_thresh', default=0.6, type=float,
                        help='NMS threshold')
    parser.add_argument('--center_sample', action='store_true', default=False,
                        help='center sample trick.')
    # dataset
    parser.add_argument('--root', default='/home/tsvanco/OD/datasets',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco-val',
                        help='voc, coco-val, coco-test.')
    # TTA
    parser.add_argument('-tta', '--test_aug', action='store_true', default=False,
                        help='use test augmentation.')
    return parser.parse_args()


def voc_test(model, data_dir, device, img_size):
    evaluator = VOCAPIEvaluator(data_root=data_dir,
                                img_size=img_size,
                                device=device,
                                transform=ValTransforms(img_size),
                                display=True
                                )

    # VOC evaluation
    evaluator.evaluate(model)


def coco_test(model, data_dir, device, img_size, test=False):
    if test:
        # test-dev
        print('test on test-dev 2017')
        evaluator = COCOAPIEvaluator(
            data_dir=data_dir,
            img_size=img_size,
            device=device,
            testset=True,
            transform=ValTransforms(img_size)
        )

    else:
        # eval
        evaluator = COCOAPIEvaluator(
            data_dir=data_dir,
            img_size=img_size,
            device=device,
            testset=False,
            transform=ValTransforms(img_size)
        )

    # COCO evaluation
    evaluator.evaluate(model)


def FLOPs_and_Params(model, img_size, device):
    x = torch.randn(1, 3, img_size, img_size).to(device)
    print('==============================')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    flop, param = flops / 1e9, params / 1e6
    print('GFLOPs : {:.2f}'.format(flop * 2))
    print('Params : {:.2f} M'.format(param))
    return flop, param


if __name__ == '__main__':
    args = parse_args()
    # dataset
    if args.dataset == 'voc':
        print('eval on voc ...')
        num_classes = 20
        data_dir = os.path.join(args.root, 'VOCdevkit')
    elif args.dataset == 'coco-val':
        print('eval on coco-val ...')
        num_classes = 80
        data_dir = os.path.join(args.root, 'COCO')
    elif args.dataset == 'coco-test':
        print('eval on coco-test-dev ...')
        num_classes = 80
        data_dir = os.path.join(args.root, 'COCO')
    else:
        print('unknow dataset !! we only support voc, coco-val, coco-test !!!')
        exit(0)

    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # YOLO Config
    cfg = yolo_config[args.model]
    # build model
    model = build_model(args=args,
                        cfg=cfg,
                        device=device,
                        num_classes=num_classes,
                        trainable=False)

    # load weight
    model.load_state_dict(torch.load(
        args.weight, map_location='cpu'), strict=False)
    model = model.to(device).eval()
    print('Finished loading model!')

    # compute FLOPs and Params
    model_copy = deepcopy(model)
    model_copy.trainable = False
    model_copy.eval()
    FLOPs_and_Params(
        model=model_copy,
        img_size=args.img_size,
        device=device)
    del model_copy

    # TTA
    test_aug = TestTimeAugmentation(
        num_classes=num_classes) if args.test_aug else None

    # evaluation
    with torch.no_grad():
        if args.dataset == 'voc':
            voc_test(model, data_dir, device, args.img_size)
        elif args.dataset == 'coco-val':
            coco_test(model, data_dir, device, args.img_size, test=False)
        elif args.dataset == 'coco-test':
            coco_test(model, data_dir, device, args.img_size, test=True)
