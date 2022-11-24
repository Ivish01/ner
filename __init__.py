
class DataGenerater:
    def __init__(self, source, x_labels=None, y_labels=None, is_csv=True, is_dir=False, txt_file=False, test_set_ratio=30, *kwargs):
        self.source = source
        self.x_labels = x_labels
        self.y_labels = y_labels
        self.load_data(is_csv=is_csv, is_dir=is_dir, txt_file=txt_file)

    def split_sets(self, *kwargs):
        pass

    def load_data(self, is_csv=True, is_dir=False, txt_file=False):
        pass

    def yield_data(self, test_set=False, size=100):
        #yield x, y
        pass
        


class PredectionModel:
    
    @staticmethod
    def init_from_config():
        pass

    def __init__(self):
        pass

    def load_model(self):
        pass

    def predect(self, y):
        pass

    def init_model(self, **kwargs):
        pass

    def train(self, train_x, train_y):
        pass
    
    def test_n_validate(self, test_x, orginal_y):
        pass

