import os, json

from argparse import ArgumentParser

args = ArgumentParser()
args.add_argument('--eval-dir', help='Directory containing video files.', required=True)
args = args.parse_args()

#TODO