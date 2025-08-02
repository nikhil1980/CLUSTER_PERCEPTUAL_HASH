#!/usr/bin/env python
"""
Image Hash:

https://github.com/JohannesBuchner/imagehash?tab=readme-ov-file

https://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

"""
import imagehash
import os
import json
import shutil
import multiprocessing
import time
import numpy as np
from PIL import Image
from alive_progress import alive_bar
from hyperparam import HYPERPARAM
import warnings

from pillow_heif import register_heif_opener

# For HEIC
register_heif_opener()


warnings.filterwarnings("ignore", category=UserWarning)


def connected_components(neighbors):
    """
    Compute connected components from an adjacency list
    Based on https://stackoverflow.com/a/13837045

    Time complexity: O(N^2)

    :param neighbors: map node -> list of neighbors
    :return: iterator over the connected components (as sets of nodes)
    """
    seen = set()

    def component(node):
        result = set()
        nodes = {node}
        while nodes:
            node = nodes.pop()
            seen.add(node)
            nodes |= neighbors[node] - seen
            result.add(node)
        return result

    for node in neighbors:
        if node not in seen:
            yield component(node)


def _hash(args):
    """
    Calculate the hash

    :param args:
    :return:
    """
    i, f, IMAGE_DIR, algo = args
    hash_fn = HYPERPARAM.ALGORITHMS[algo][0]
    if i % 100 == 0:
        print(i, f)
    i = Image.open(os.path.join(os.path.join(HYPERPARAM.dataset_path, IMAGE_DIR), f))
    hash = hash_fn(i)
    return f, hash.hash.tolist()


def go(algorithm, IMAGE_DIR):
    """
    Compute hashes (and store) them, and then run clustering for a particular hash algorithm

    :param algorithm: the key for the algorithm to run (@see HYPERPARAM.ALGORITHMS)
    :param IMAGE_DIR: input folder to images

    """
    start_time = time.time()
    tmp = os.path.join(HYPERPARAM.output_path, IMAGE_DIR)
    output = os.path.join(tmp, algorithm)
    os.makedirs(output, exist_ok=True)
    print(f"Running ALGO: {algorithm} on DIR: {IMAGE_DIR} and writing to OUTPUT: {output}")

    hashfn, max_threshold, threshold_step = HYPERPARAM.ALGORITHMS[algorithm]

    hashes = {}  # Need to cache

    hashes_json_path = os.path.join(output, 'hashes.json')
    if os.path.exists(hashes_json_path):
        # Hash file is created but needs to update with new images
        print(f"Loading hashes from: {hashes_json_path}")
        with open(hashes_json_path) as inf:
            data = json.load(inf)
        for f, h in data.items():
            hashes[f] = imagehash.ImageHash(np.array(h))
    else:
        # Hash file never got created
        pool = multiprocessing.Pool(processes=max(1, os.cpu_count() - 1))
        args = [(i, f, IMAGE_DIR, algorithm) for i, f in enumerate(sorted(os.listdir(os.path.join(HYPERPARAM.dataset_path, IMAGE_DIR))))]
        result = pool.imap_unordered(_hash, args)
        hashes = {f: imagehash.ImageHash(np.array(h)) for f, h in result}

    files = set(hashes.keys())
    hashes_sorted = sorted(hashes.items(), key=lambda h: str(h[1]))

    print(f'\nHashes computed in {time.time() - start_time: .2f} seconds')

    hashes_to_write = {}
    for f, h in hashes_sorted:
        hashes_to_write[f] = h.hash.tolist()
    with open(hashes_json_path, 'w') as outf:
        json.dump(hashes_to_write, outf)

    print(f'\nHashes persisted in time: {time.time() - start_time: .2f} seconds')

    diffs = []
    for i1, h1 in enumerate(hashes_sorted):
        if i1 % 100 == 0:
            print('diff', i1, ' time ', time.time() - start_time)
        for i2, h2 in enumerate(hashes_sorted):
            if i1 < i2:
                diff = int(h1[1] - h2[1]) / len(str(h1[1]))
                if diff < max_threshold:
                    diffs.append((diff, h1[0], h2[0]))

    # diffs = sorted(diffs, key=lambda d: d[0])
    print(diffs)
    print(f'\nHash Diffs computed in time: {time.time() - start_time: .2f} seconds')

    # with open(os.path.join(output, 'diffs.txt'), 'w') as outf:
    #     for d in diffs:
    #         print(d, file=outf)

    # print('Min diffs:\n', '\n'.join(str(d) for d in diffs[:20]))
    # print('Max diffs:\n', '\n'.join(str(d) for d in diffs[-20:]))
    # print('Avg diff: ', sum(d[0] for d in diffs) / len(diffs))

    for threshold in range(0, max_threshold, threshold_step):
        print(f'Copying for threshold: {threshold} in time: {time.time() - start_time} seconds')

        neighbors = {}
        for d in diffs:
            if d[0] <= threshold:
                neighbors.setdefault(d[1], set()).add(d[2])
                neighbors.setdefault(d[2], set()).add(d[1])

        # Sort clusters based on membership cardinality
        clusters = list(sorted(
            connected_components(neighbors),
            key=lambda c: len(c),
            reverse=True))

        print(clusters)
        in_clusters = set().union(*clusters)
        unclustered = files - in_clusters

        destdir = os.path.join(output, 'thr_{}_unclustered_{}'.format(
            str(threshold).zfill(3), len(unclustered)))
        shutil.rmtree(destdir, ignore_errors=True)
        os.makedirs(destdir)

        for cnt, cluster in enumerate(clusters + [unclustered]):
            if cnt == len(clusters):
                name = 'unclustered'
            else:
                name = str(cnt + 1).zfill(3)
            cdir = os.path.join(destdir, '{}_{}'.format(name, len(cluster)))
            os.makedirs(cdir)
            for f in cluster:
                fname, ext = os.path.splitext(f)
                if len(str(hashes[f])) > 128:
                    filename = '{}_{}{}'.format(fname, str(hashes[f])[:120], ext)
                else:
                    filename = '{}_{}{}'.format(fname, str(hashes[f]), ext)

                os.symlink(os.path.abspath(os.path.join(os.path.join(HYPERPARAM.dataset_path, IMAGE_DIR), f)), os.path.join(cdir, filename))

    end_time = time.time()

    print('Time taken: ', round(end_time - start_time, 2), 'sec\n\n')


def main():
    list_of_dir = sorted(os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                 HYPERPARAM.dataset_path)))

    # Remove .DS_STORE
    if '.DS_Store' in list_of_dir:
        list_of_dir.remove('.DS_Store')

    with alive_bar(len(list_of_dir), bar='halloween', spinner='notes', title='SSI', force_tty=True) as bar:
        for dir_name in list_of_dir:
            print(f"Processing: {dir_name}")
            for algorithm in HYPERPARAM.ALGORITHMS.keys():
                go(algorithm, dir_name)
            bar()


if __name__ == '__main__':
    """
    1. Create all base folders if they don't exist
    """
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             HYPERPARAM.dataset_path)
    os.makedirs(path, exist_ok=True)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             HYPERPARAM.log_path)
    os.makedirs(path, exist_ok=True)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             HYPERPARAM.output_path)
    os.makedirs(path, exist_ok=True)

    """
    2. Call main
    """
    main()
