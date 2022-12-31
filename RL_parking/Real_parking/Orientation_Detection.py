from Car_Direction import cnn, method, dataset, tools
import numpy as np


TEST_PATH = "Labels/direction_label.txt"
test_dataset = dataset.Dataset(TEST_PATH,batch_size=1,mode='test')
prs = method.test('Models/orientation_detection_model.pdparams', test_dataset)
print(prs)
