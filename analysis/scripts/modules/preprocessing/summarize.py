import re
import pickle
from pyprojroot import here
import os
from pathlib import Path
os.chdir(here() / Path('sandbox'))
script_root = Path('scripts/preprocess')


with open(script_root / Path('log.txt'), 'r') as file:
    log = file.read()

pattern = r'Error with (sub.*bld\d+)\.'
log_parse = re.findall(pattern, log)

with open(script_root / Path('log_condensed.txt'), 'w') as file:
    for line in log_parse:
        file.write(str(line) + '\n')

file.close()


with open('data/formatted/dimitrios.pkl', 'rb') as file:
    data = pickle.load(file)



out = []

for subject in data:
    row = f'\nSubject: {subject}:\n'
    for session in data[subject]:
        row += f'Session: {session}\n'
        row += f'Runs: {len(data[subject][session].keys())}'
        out.append(row)
        row = ''


with open(script_root / Path('data_summary.txt'), 'w') as file:
    for line in out:
        file.write('\n')
        file.write(line)

file.close()

