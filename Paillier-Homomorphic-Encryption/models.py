import numpy as np


def encrypt_vector(public_key, x):
    return [public_key.encrypt(i) for i in x]


def encrypt_matrix(public_key, x):
    ret = []
    for r in x:
        ret.append(encrypt_vector(public_key, r))
    return ret


def decrypt_vector(private_key, x):
    return [private_key.decrypt(i) for i in x]


def decrypt_matrix(private_key, x):
    ret = []
    for r in x:
        ret.append(decrypt_vector(private_key, r))
    return ret


class LRModel(object):

    def __init__(self, public_key, w_size=None, w=None, encrypted=False):
        self.public_key = public_key
        if w is not None:
            self.weights = w
        else:
            self.weights = np.random.uniform(-0.5, 0.5, (w_size,))

        if not encrypted:
            self.encrypt_weights = encrypt_vector(public_key, self.weights)
        else:
            self.encrypt_weights = self.weights

    def set_encrypt_weights(self, w):
        for id, e in enumerate(w):
            self.encrypt_weights[id] = e

    def set_raw_weights(self, w):
        for id, e in enumerate(w):
            self.weights[id] = e
