#!/usr/bin/env python2
import subprocess
import argparse
import os.path
import os
import re
from collections import deque


compute_flow_path = os.path.join(os.path.dirname(__file__), 'compute_flow.py')
number_pattern = re.compile('frame_(\d+).jpg')


def parse_frame_number(frame_filename):
    match = number_pattern.match(os.path.basename(frame_filename))
    return int(match.group(1))


parser = argparse.ArgumentParser()
parser.add_argument('frame_dir', type=str)
parser.add_argument('flow_dir', type=str)
parser.add_argument('--frame_pattern', type=str, default="frame_.*.jpg")
parser.add_argument('--stride', default=2, type=int)
parser.add_argument('--bound', default=25, type=float)
parser.add_argument('--dilation', default=3, type=int)

args = parser.parse_args()

frame_pattern = re.compile(args.frame_pattern)
frame_dir = args.frame_dir
flow_dir = args.flow_dir

dilation = args.dilation
bound = args.bound
stride = args.stride

assert bound >= 0
assert stride >= 1
assert dilation >= 1


frames = sorted([os.path.join(frame_dir, p) for p in os.listdir(frame_dir) if frame_pattern.match(p)])
assert frames >= dilation
frames_queue = deque(frames[:dilation])
frames = deque(frames[dilation:])

if not os.path.exists(flow_dir):
    os.makedirs(flow_dir)

while True:
    assert len(frames_queue) == dilation
    frame_0 = frames_queue.popleft()
    frame_1 = frames_queue.pop()
    frame_0_index = parse_frame_number(frame_0)
    flow_u = os.path.join(flow_dir, 'flow_x_{:010d}.jpg'.format(frame_0_index))
    flow_v = os.path.join(flow_dir, 'flow_y_{:010d}.jpg'.format(frame_0_index))
    print ".",

    cli_args = [
        "python", compute_flow_path,
        "--bound", str(bound),
        frame_0, frame_1, flow_u, flow_v
    ]
    print(cli_args)
    result = subprocess.check_call(cli_args)


    try:
        frames_queue.append(frame_1)
        for i in range(0, stride):
            if i > 0:
                frames_queue.pop()
            frames_queue.append(frames.popleft())
    except IndexError:
        break
