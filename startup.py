# -*- coding:utf-8 -*-
from gbdt.data import DataSet
from gbdt.model import GBDT
from math import exp, log

__author__ = 'jun yuan'


def train_model():
    data_file = './data/feature_data.csv'
    dateset = DataSet(data_file)

    gbdt = GBDT(max_iter=80, sample_rate=0.8, learn_rate=0.1, max_depth=7, loss_type='regression')
    gbdt.fit(dateset, set(list(dateset.get_instances_idset())[:1200]))

    GBDT.save_model(gbdt, "./", "test")

    predict = gbdt.predict(dateset.instances[1])
    print "predict", predict, dateset.get_instance(1)['label']
    print "#########################"
    predict = gbdt.predict(dateset.instances[2])
    print "predict", predict, dateset.get_instance(2)['label']
    print "#########################"
    predict = gbdt.predict(dateset.instances[3])
    print "predict", predict, dateset.get_instance(3)['label']
    print "#########################"
    predict = gbdt.predict(dateset.instances[4])
    print "predict", predict, dateset.get_instance(4)['label']
    predict = gbdt.predict(dateset.instances[402])
    print "predict", predict, dateset.get_instance(402)['label']


def test_model():
    data_file = './data/feature_data.csv'
    dateset = DataSet(data_file)
    gbdt = GBDT.load_model("./", "test")
    loss = 0.0
    err_num = 0
    for item_id in dateset.get_instances_idset():
        predict = gbdt.predict(dateset.instances[item_id])
        print "predict", predict, dateset.get_instance(item_id)['label']

        y_i = dateset.get_instance(item_id)['label']
        f_value = predict
        p_1 = 1 / (1 + exp(-2 * f_value))
        loss -= ((1 + y_i) * log(p_1) / 2) + ((1 - y_i) * log(1 - p_1) / 2)
        if abs(y_i - predict) > 0.5:
            err_num += 1
    print "loss %s" % (loss/dateset.size())
    print "err_classifier rate: %s" % ((err_num*1.0)/dateset.size())

if __name__ == '__main__':
    train_model()
    test_model()




