import pprint as pp

class remapping():
    def __init__(self):
        self.fine_labels = [
            'apple',  # id 0
            'aquarium_fish',
            'baby',
            'bear',
            'beaver',
            'bed',
            'bee',
            'beetle',
            'bicycle',
            'bottle',
            'bowl',
            'boy',
            'bridge',
            'bus',
            'butterfly',
            'camel',
            'can',
            'castle',
            'caterpillar',
            'cattle',
            'chair',
            'chimpanzee',
            'clock',
            'cloud',
            'cockroach',
            'couch',
            'crab',
            'crocodile',
            'cup',
            'dinosaur',
            'dolphin',
            'elephant',
            'flatfish',
            'forest',
            'fox',
            'girl',
            'hamster',
            'house',
            'kangaroo',
            'computer_keyboard',
            'lamp',
            'lawn_mower',
            'leopard',
            'lion',
            'lizard',
            'lobster',
            'man',
            'maple_tree',
            'motorcycle',
            'mountain',
            'mouse',
            'mushroom',
            'oak_tree',
            'orange',
            'orchid',
            'otter',
            'palm_tree',
            'pear',
            'pickup_truck',
            'pine_tree',
            'plain',
            'plate',
            'poppy',
            'porcupine',
            'possum',
            'rabbit',
            'raccoon',
            'ray',
            'road',
            'rocket',
            'rose',
            'sea',
            'seal',
            'shark',
            'shrew',
            'skunk',
            'skyscraper',
            'snail',
            'snake',
            'spider',
            'squirrel',
            'streetcar',
            'sunflower',
            'sweet_pepper',
            'table',
            'tank',
            'telephone',
            'television',
            'tiger',
            'tractor',
            'train',
            'trout',
            'tulip',
            'turtle',
            'wardrobe',
            'whale',
            'willow_tree',
            'wolf',
            'woman',
            'worm',
        ]

        self.mapping_coarse_fine = {
            'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
            'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
            'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
            'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
            'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear',
                                    'sweet_pepper'],
            'household electrical device': ['clock', 'computer_keyboard', 'lamp',
                                            'telephone', 'television'],
            'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
            'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
            'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
            'large man-made outdoor things': ['bridge', 'castle', 'house', 'road',
                                            'skyscraper'],
            'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain',
                                            'sea'],
            'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee',
                                            'elephant', 'kangaroo'],
            'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
            'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
            'people': ['baby', 'boy', 'girl', 'man', 'woman'],
            'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
            'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
            'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree',
                    'willow_tree'],
            'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
            'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
        }

        self.fine_id, self.id_fine, self.coarse_id, self.id_coarse, self.fine_coarse, self.fine_id_coarse_id, self.coarse_id_fine_id = self._define_dicts()


    def print_fine_labels(self):
        for id, label in enumerate(self.fine_labels):
            print(id, " ", label)
    
    def print_dicts(self):
        print('fine_id:\n', self.fine_id)
        print('id_fine:\n', self.id_fine)
        print('coarse_id:\n', self.coarse_id)
        print('id_coarse:\n', self.id_coarse)
        print('fine_coarse:\n', self.fine_coarse)
        print('fine_id_coarse_id:\n', self.fine_id_coarse_id)
        print('coarse_id_fine_id:\n', self.coarse_id_fine_id)



    def _define_dicts(self):
        # fine label name -> id of fine label
        fine_id = dict()
        # id of fine label -> fine label name
        id_fine = dict()
        for id, label in enumerate(self.fine_labels):
            fine_id[label] = id
            id_fine[id] = label

        # coarse label name -> id of coarse label
        coarse_id = dict()
        # id of coarse label -> name of the coarse label
        id_coarse = dict()
        # name of fine label -> name of coarse label
        fine_coarse = dict()
        # id of fine label -> id of coarse label
        fine_id_coarse_id = dict()
        # id of coarse label -> id of fine label
        coarse_id_fine_id = dict()
        for id, (coarse, fines) in enumerate(self.mapping_coarse_fine.items()):
            coarse_id[coarse] = id
            id_coarse[id] = coarse
            fine_labels_ids = []
            for fine in fines:
                fine_coarse[fine] = coarse
                fine_label_id = fine_id[fine]
                fine_id_coarse_id[fine_label_id] = id
                fine_labels_ids.append(fine_label_id)
            coarse_id_fine_id[id] = fine_labels_ids

        
        return fine_id, id_fine, coarse_id, id_coarse, fine_coarse, fine_id_coarse_id, coarse_id_fine_id


if __name__ == "__main__":
    r = remapping()
    r.print_fine_labels()
    pp.pprint(r.mapping_coarse_fine)
    r.print_dicts()
