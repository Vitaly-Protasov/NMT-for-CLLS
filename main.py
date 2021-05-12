from easynmt import EasyNMT
import torch
import os
import transformers
import itertools
import re
from typing import List, Tuple
import spacy
import pandas as pd
from tqdm import tqdm
import subprocess
from copy import deepcopy


