CLASSES = {'pascal': ['background', 'road', 'farmland', 'building', 'water', 'grassland', 'bus',
                      'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike', 
                      'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor',
                      'ceiling-tile', 'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
                      'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble', 'floor-other','floor-other2',
                      ],
          'pascal1': ['background', 'road', 'farmland', 'building', 'water', 'grassland', 'bus',
                      'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike',
                      'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor',
                      'ceiling-tile', 'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
                      'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble', 'floor-other','floor-other2',
                      ],
           'hw': [
               "background",  # 0: 背景类别
               "water",  # 1: 水体
               "road",  # 2: 道路
               "building",  # 3: 建筑物
               "airport",  # 4: 机场
               "train_station",  # 5: 火车站
               "solar_panel",  # 6: 光伏
               "parking_lot",  # 7: 停车场
               "playground",  # 8: 操场
               "farmland",  # 9: 普通耕地
               "greenhouse",  # 10: 农业大棚
               "natural_grass",  # 11: 自然草地
               "green_space",  # 12: 绿地绿化
               "natural_forest",  # 13: 自然林
               "artificial_forest",  # 14: 人工林
               "natural_bare_soil",  # 15: 自然裸土
               "artificial_bare_soil",  # 16: 人为裸土
               "other"  # 17: 其它
           ],

           'cityscapes': ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
                          'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                          'truck', 'bus', 'train', 'motorcycle', 'bicycle',],
	   'cityscapes1': ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
                          'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                          'truck', 'bus', 'train', 'motorcycle', 'bicycle',],
           
           'coco': ['void', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
                    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 
                    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
                    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 
                    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 
                    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
                    'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'branch', 'bridge', 
                    'building-other', 'bush', 'cabinet', 'cage', 'cardboard', 'carpet', 'ceiling-other', 
                    'ceiling-tile', 'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
                    'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble', 'floor-other', 'floor-stone', 
                    'floor-tile', 'floor-wood', 'flower', 'fog', 'food-other', 'fruit', 'furniture-other', 
                    'grass', 'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal', 
                    'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper', 'pavement', 'pillow', 
                    'plant-other', 'plastic', 'platform', 'playingfield', 'railing', 'railroad', 'river', 
                    'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper',
                    'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other', 'table', 'tent',
                    'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick', 'wall-concrete', 'wall-other', 
                    'wall-panel', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
                    'window-blind', 'window-other', 'wood'],
           }