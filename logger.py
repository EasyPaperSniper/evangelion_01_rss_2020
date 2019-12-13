from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import json
import os
import shutil
import torch
import torchvision
import numpy as np
from termcolor import colored


FORMAT_CONFIG = {
    'step_0': {
        'train': [('iterations', 'iter', 'int'),
                  ('model_loss', 'MLOSS', 'float'),],
        'eval': [('step', 'S', 'int'),
                 ('x_pos', 'x', 'float'),
                 ('y_pos', 'y', 'float'),
                 ('x_vel', 'vx', 'float'),
                 ('y_vel', 'vy', 'float'),],
        'tgt': [('step', 'S', 'int'),
                 ('x_pos', 'x', 'float'),
                 ('y_pos', 'y', 'float'),
                 ('x_vel', 'vx', 'float'),
                 ('y_vel', 'vy', 'float'),]
    }
}

class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, file_name, formating):
        self._file_name = file_name
        if os.path.exists(file_name):
            os.remove(file_name)
        self._formating = formating
        self._meters = defaultdict(AverageMeter)

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith('train'):
                key = key[len('train') + 1:]
            elif key.startswith('eval'):
                key = key[len('eval') + 1:]
            else:
                key = key[len('tgt') + 1:]
            key = key.replace('/', '_')
            data[key] = meter.value()
        return data

    def _dump_to_file(self, data):
        with open(self._file_name, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def _format(self, key, value, ty):
        template = '%s: '
        if ty == 'int':
            template += '%d'
        elif ty == 'float':
            template += '%.04f'
        elif ty == 'time':
            template += '%.01f s'
        else:
            raise 'invalid format type: %s' % ty
        return template % (key, value)

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        pieces = ['{:5}'.format(prefix)]
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print('| %s' % (' | '.join(pieces)))

    def dump(self, step, prefix, console = False):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data['steps'] = step
        self._dump_to_file(data)
        if console:
            self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(self, log_dir, name, config='step_0'):
        self.name = name
        self._log_dir = log_dir
        self.mg = MetersGroup(
            os.path.join(log_dir, self.name + '.log'),
            formating=FORMAT_CONFIG[config][self.name])

    def log(self, key, value, n=1):
        if type(value) == torch.Tensor:
            value = value.item()
        mg = self.mg
        mg.log(key, value, n)

    def dump(self, step, console=False):
        self.mg.dump(step, self.name, console)

