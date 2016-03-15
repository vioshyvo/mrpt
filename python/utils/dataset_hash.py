# -*- coding: utf-8 -*-
#
# Source:
# http://stackoverflow.com/questions/806151/how-to-hash-a-large-object-dataset-in-python/806342#806342

import hashlib as hl
import numpy as np


def hash_data(data):
    return hl.sha1(data.view(np.uint8)).hexdigest()
