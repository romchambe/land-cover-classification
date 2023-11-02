from utils.train import train
from utils.preprocess import preprocess_data
import sys

if len(sys.argv) > 1 and sys.argv[1] == '--should-preprocess':
    preprocess_data()

train()
