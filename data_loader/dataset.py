# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:54
# @Author  : zhoujun
import pathlib
import os
import cv2
import json
import numpy as np
import scipy.io as sio
from tqdm.auto import tqdm

from base import BaseDataSet
from utils import order_points_clockwise, get_datalist, load, expand_polygon


class ICDAR2015Dataset(BaseDataSet):
    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, ignore_tags, transform=None, **kwargs):
        super().__init__(data_path, img_mode, pre_processes, filter_keys, ignore_tags, transform)

    def load_data(self, data_path: str) -> list:
        data_list = get_datalist(data_path)
        t_data_list = []
        for img_path, label_path in data_list:
            data = self._get_annotation(label_path)
            if len(data['text_polys']) > 0:
                item = {'img_path': img_path, 'img_name': pathlib.Path(img_path).stem}
                item.update(data)
                t_data_list.append(item)
            else:
                print('there is no suit bbox in {}'.format(label_path))
        return t_data_list

    def _get_annotation(self, label_path: str) -> dict:
        boxes = []
        texts = []
        ignores = []
        with open(label_path, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                try:
                    box = order_points_clockwise(np.array(list(map(float, params[:8]))).reshape(-1, 2))
                    if cv2.contourArea(box) > 0:
                        boxes.append(box)
                        label = params[8]
                        texts.append(label)
                        ignores.append(label in self.ignore_tags)
                except:
                    print('load label failed on {}'.format(label_path))
        data = {
            'text_polys': np.array(boxes),
            'texts': texts,
            'ignore_tags': ignores,
        }
        return data


class DetDataset(BaseDataSet):
    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, ignore_tags, transform=None, **kwargs):
        self.load_char_annotation = kwargs['load_char_annotation']
        self.expand_one_char = kwargs['expand_one_char']
        super().__init__(data_path, img_mode, pre_processes, filter_keys, ignore_tags, transform)

    def load_data(self, data_path: str) -> list:
        """
        从json文件中读取出 文本行的坐标和gt，字符的坐标和gt
        :param data_path:
        :return:
        """
        data_list = []
        for path in data_path:
            content = load(path)
            for gt in tqdm(content['data_list'], desc='read file {}'.format(path)):
                img_path = os.path.join(content['data_root'], gt['img_name'])
                polygons = []
                texts = []
                illegibility_list = []
                language_list = []
                for annotation in gt['annotations']:
                    if len(annotation['polygon']) == 0 or len(annotation['text']) == 0:
                        continue
                    if len(annotation['text']) > 1 and self.expand_one_char:
                        annotation['polygon'] = expand_polygon(annotation['polygon'])
                    polygons.append(annotation['polygon'])
                    texts.append(annotation['text'])
                    illegibility_list.append(annotation['illegibility'])
                    language_list.append(annotation['language'])
                    if self.load_char_annotation:
                        for char_annotation in annotation['chars']:
                            if len(char_annotation['polygon']) == 0 or len(char_annotation['char']) == 0:
                                continue
                            polygons.append(char_annotation['polygon'])
                            texts.append(char_annotation['char'])
                            illegibility_list.append(char_annotation['illegibility'])
                            language_list.append(char_annotation['language'])
                data_list.append({'img_path': img_path, 'img_name': gt['img_name'], 'text_polys': np.array(polygons),
                                  'texts': texts, 'ignore_tags': illegibility_list})
        return data_list


class BILLDataset(BaseDataSet):
    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, ignore_tags, transform=None, **kwargs):
        """
        Load labelme format dataset!
        :param data_path:
        :param img_mode:
        :param pre_processes:
        :param filter_keys:
        :param ignore_tags:
        :param transform:
        :param kwargs:
        """
        self.load_char_annotation = kwargs['load_char_annotation']
        self.expand_one_char = kwargs['expand_one_char']
        super().__init__(data_path, img_mode, pre_processes, filter_keys, ignore_tags, transform)

    def load_data(self, data_path: str) -> list:
        """
        从json文件中读取出 文本行的坐标和gt，字符的坐标和gt
        :param data_path:
        :return:
        """
        data_list = []
        json_list, image_list = self._check_json(data_path[0])
        for json_path, img_path in tqdm(zip(json_list, image_list), desc='search file in {}'.format(data_path)):
            polygons = []
            texts = []
            illegibility_list = []
            language_list = []

            info = json.load(open(json_path, 'r'))
            for i, obj in enumerate(info['shapes']):
                poly = obj['points']
                if len(poly) == 2:
                    [xmin, ymin], [xmax, ymax] = poly[0], poly[1]
                    poly = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]

                text = obj['label']
                if len(poly) == 0 or len(text) == 0:
                    continue
                if len(text) > 1 and self.expand_one_char:
                    poly = expand_polygon(poly)
                polygons.append(poly)
                texts.append(text)
                illegibility_list.append(False)
                language_list.append('EN')
                # if self.load_char_annotation:
                #     for char_annotation in annotation['chars']:
                #         if len(char_annotation['polygon']) == 0 or len(char_annotation['char']) == 0:
                #             continue
                #         polygons.append(char_annotation['polygon'])
                #         texts.append(char_annotation['char'])
                #         illegibility_list.append(char_annotation['illegibility'])
                #         language_list.append(char_annotation['language'])
            data_list.append({'img_path': img_path, 'img_name': os.path.basename(img_path),
                              'text_polys': np.array(polygons), 'texts': texts,
                              'ignore_tags': illegibility_list})
        print('Got {} useful samples!'.format(len(data_list)))
        return data_list

    def _check_json(self, data_path):
        """ Check the format of json file and images to avoid going deep!
        Ignore the none-instance sample!
         """
        json_list, image_list = [], []
        assert os.path.isdir(data_path), '{} is not a valid directory'.format(data_path)

        for root, _, fnames in sorted(os.walk(data_path)):
            for fname in sorted(fnames):
                if fname.endswith('.json'):
                    json_path = os.path.join(root, fname)
                    info = json.load(open(json_path, 'r'))
                    img_path = os.path.join(root, info['imagePath'])
                    img = cv2.imread(img_path)
                    if img is None or 'shapes' not in info.keys():
                        continue
                    # Ignore no-instance sample!
                    Got = False
                    for i, obj in enumerate(info['shapes']):
                        poly = obj['points']
                        if len(poly) not in [2, 4]:
                            print('Ignore polygons of point number not in [2, 4]!')
                            continue
                        Got = True
                    if Got:
                        json_list.append(json_path)
                        image_list.append(img_path)

        return json_list, image_list


class SynthTextDataset(BaseDataSet):
    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, transform=None, **kwargs):
        self.transform = transform
        self.dataRoot = pathlib.Path(data_path)
        if not self.dataRoot.exists():
            raise FileNotFoundError('Dataset folder is not exist.')

        self.targetFilePath = self.dataRoot / 'gt.mat'
        if not self.targetFilePath.exists():
            raise FileExistsError('Target file is not exist.')
        targets = {}
        sio.loadmat(self.targetFilePath, targets, squeeze_me=True, struct_as_record=False,
                    variable_names=['imnames', 'wordBB', 'txt'])

        self.imageNames = targets['imnames']
        self.wordBBoxes = targets['wordBB']
        self.transcripts = targets['txt']
        super().__init__(data_path, img_mode, pre_processes, filter_keys, transform)

    def load_data(self, data_path: str) -> list:
        t_data_list = []
        for imageName, wordBBoxes, texts in zip(self.imageNames, self.wordBBoxes, self.transcripts):
            item = {}
            wordBBoxes = np.expand_dims(wordBBoxes, axis=2) if (wordBBoxes.ndim == 2) else wordBBoxes
            _, _, numOfWords = wordBBoxes.shape
            text_polys = wordBBoxes.reshape([8, numOfWords], order='F').T  # num_words * 8
            text_polys = text_polys.reshape(numOfWords, 4, 2)  # num_of_words * 4 * 2
            transcripts = [word for line in texts for word in line.split()]
            if numOfWords != len(transcripts):
                continue
            item['img_path'] = str(self.dataRoot / imageName)
            item['img_name'] = (self.dataRoot / imageName).stem
            item['text_polys'] = text_polys
            item['texts'] = transcripts
            item['ignore_tags'] = [x in self.ignore_tags for x in transcripts]
            t_data_list.append(item)
        return t_data_list


if __name__ == '__main__':
    import torch
    import anyconfig
    from torch.utils.data import DataLoader
    from torchvision import transforms

    from utils import parse_config, show_img, plt, draw_bbox

    config = anyconfig.load('config/icdar2015_resnet18_FPN_DBhead_polyLR.yaml')
    config = parse_config(config)
    dataset_args = config['dataset']['train']['dataset']['args']
    # dataset_args.pop('data_path')
    # data_list = [(r'E:/zj/dataset/icdar2015/train/img/img_15.jpg', 'E:/zj/dataset/icdar2015/train/gt/gt_img_15.txt')]
    train_data = ICDAR2015Dataset(data_path=dataset_args.pop('data_path'), transform=transforms.ToTensor(),
                                  **dataset_args)
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True, num_workers=0)
    for i, data in enumerate(tqdm(train_loader)):
        # img = data['img']
        # shrink_label = data['shrink_map']
        # threshold_label = data['threshold_map']
        #
        # print(threshold_label.shape, threshold_label.shape, img.shape)
        # show_img(img[0].numpy().transpose(1, 2, 0), title='img')
        # show_img((shrink_label[0].to(torch.float)).numpy(), title='shrink_label')
        # show_img((threshold_label[0].to(torch.float)).numpy(), title='threshold_label')
        # img = draw_bbox(img[0].numpy().transpose(1, 2, 0),np.array(data['text_polys']))
        # show_img(img, title='draw_bbox')
        # plt.show()
        pass
