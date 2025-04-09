#!/usr/bin/python
import sys, yaml
from simlar.utils import *

cmd='list'

if len(sys.argv) > 1:
    cmd = sys.argv[1].lower()

if not cmd in ['list','dump']:
    print("Usage: simlar-config.py [list|dump]")
    sys.exit(1)

if cmd == 'list':
    print('Listing available configuration keywords:')
    for cfg in list_config():
        print(cfg)

elif cmd == 'dump':
    print('Dumping available configuration contents:')
    for cfg in list_config():
        print(f"\nConfiguration: {cfg}")
        config = load_config(cfg)
        print(yaml.dump(config, default_flow_style=False))

elif cmd == 'file':
    print('Reading a used configuration from simlar output file')
    if len(sys.argv) < 3:
        print("Usage: simlar-config.py file <filename>")
        sys.exit(1)
    filename = sys.argv[2]
    with h5.File(filename,'r') as fin:
        config = json.loads(fin.attrs['config'])
        print(yaml.dump(config, default_flow_style=False))

        