class DefaultConfig(object):

    num_classes = 2
    env = 'default'
    model = 'AlexNet'

    train_data_root = '/home/jzx/Project/DL/DATA/DogsCats/orgin_train'
    test_data_root = '/home/jzx/Project/DL/DATA/DogsCats/orgin_train'
    load_model_path = None

    batch_size = 128  # batch size
    use_gpu = False  # use GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 10
    lr = 0.1  # initial learning rate
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4

    def parse(self, kwargs):
        import warnings
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" % k)
            setattr(self, k, v)
        print('user config: ')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))
