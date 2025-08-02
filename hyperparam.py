import torch
import imagehash


class HYPERPARAM:
    dataset_path = 'dataset'
    log_path = 'log'
    output_path = 'output'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ALGORITHMS = {
        'phash_8': (lambda i: imagehash.phash(i, hash_size=8), 2, 1),
        'avhash_8': (lambda i: imagehash.average_hash(i, hash_size=8), 2, 1),
        'dhash_8': (lambda i: imagehash.dhash(i, hash_size=8), 2, 1),
        'whash_8': (lambda i: imagehash.whash(i, hash_size=8), 2, 1),
        'phash_12': (lambda i: imagehash.phash(i, hash_size=12), 30, 2),
        'avhash_12': (lambda i: imagehash.average_hash(i, hash_size=12), 30, 2),
        'dhash_12': (lambda i: imagehash.dhash(i, hash_size=12), 30, 2),
        'phash_16': (lambda i: imagehash.phash(i, hash_size=16), 110, 3),
        'avhash_16': (lambda i: imagehash.average_hash(i, hash_size=16), 110, 3),
        'dhash_16': (lambda i: imagehash.dhash(i, hash_size=16), 110, 3),
        'whash_16': (lambda i: imagehash.whash(i, hash_size=16), 110, 3),
        'phash_20': (lambda i: imagehash.phash(i, hash_size=20), 200, 5),
        'avhash_20': (lambda i: imagehash.average_hash(i, hash_size=20), 200, 5),
        'dhash_20': (lambda i: imagehash.dhash(i, hash_size=20), 200, 5),
        'whash_32': (lambda i: imagehash.whash(i, hash_size=32), 200, 5),
    }
