import os
import argparse
import torch
import torch.nn as nn
import sys
import numpy as np
import time
import yaml

# Append root directory to system path for imports
repo_path, _ = os.path.split(os.path.realpath(__file__))
repo_path, _ = os.path.split(repo_path)
sys.path.append(repo_path)

from utils.seed import seed_all
from utils.config import CFG
from utils.dataset import get_dataset
from utils.model import get_model
from utils.logger import get_logger
from utils.io_tools import dict_to, _create_directory
import utils.checkpoint as checkpoint

def test(model, dset, _cfg, logger, out_path_root):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Moving optimizer and model to used device
    model = model.to(device=device)
    logger.info('=> Passing the network on the test set...')
    model.eval()
    inv_remap_lut = dset.dataset.get_inv_remap_lut()
    time_list = []

    with torch.no_grad():

        for t, (data, indices) in enumerate(dset):

            data = dict_to(data, device)

            # torch.cuda.synchronize()
            start_time = time.time()
            scores = model(data)  # [b,20,256,256,32]
            # torch.cuda.synchronize()
            
            time_list.append(time.time() - start_time)
            for key in scores:
                scores[key] = torch.argmax(scores[key], dim=1).data.cpu().numpy()

            curr_index = 0
            for score in scores['pred_semantic_1_1']:
                score = score.reshape(-1).astype(np.uint16)
                score = inv_remap_lut[score].astype(np.uint16)
                input_filename = dset.dataset.filepaths['occupancy'][indices[curr_index]]
                filename, extension = os.path.splitext(os.path.basename(input_filename))
                sequence = os.path.dirname(input_filename).split('/')[-2]
                out_filename = os.path.join(out_path_root, 'sequences', sequence, 'predictions', filename + '.label')
                _create_directory(os.path.dirname(out_filename))
                score.tofile(out_filename)
                logger.info('=> Sequence {} - File {} saved'.format(sequence, os.path.basename(out_filename)))
                curr_index += 1

    return time_list


def main(weights_f, out_path_root):

    # https://github.com/pytorch/pytorch/issues/27588
    torch.backends.cudnn.enabled = False

    seed_all(0)

    assert os.path.isfile(weights_f), '=> No file found at {}'

    with open('/media/max/GAME/MA/SSC-RS/cfgs/DESC.yaml', 'r') as file:
        config_dict = yaml.safe_load(file)


    # Read train configuration file
    _cfg = CFG()
    _cfg.from_dict(config_dict)
    # Setting the logger to print statements and also save them into logs file
    logger = get_logger(out_path_root, 'logs_infer.log')

    logger.info('============ Test weights: "%s" ============\n' % weights_f)
    dataset = get_dataset(_cfg._dict)['infer']

    logger.info('=> Loading network architecture...')
    model = get_model(_cfg._dict, phase='test')
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model = model.module

    logger.info('=> Loading network weights...')
    model = checkpoint.load_model(model, weights_f, logger)

    time_list = test(model, dataset, _cfg, logger, out_path_root)

    logger.info('=> ============ Network Test Done ============')

    logger.info('Inference time per frame is %.4f seconds\n' % (np.sum(time_list) / len(dataset.dataset)))

    exit()


if __name__ == '__main__':
    weights_f = "/media/max/GAME/MA/SSC-RS/ckpt/weights_epoch_035.pth"
    out_path_root = "/media/max/GAME/MA/SSC-RS/output"

    main(weights_f, out_path_root)
