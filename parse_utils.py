import os
import configparser
from pathlib import Path
from datetime import datetime
from ast import literal_eval
from collections import namedtuple


NO_PARSE = ['name']


def parse_ini(config_path: str):
    read_config = configparser.ConfigParser()
    read_config.read(config_path)

    config_attribs = []
    data_dict = {}
    print_format = {}

    for section in read_config.sections():      
        print_format[section] = []

        for (key, value) in read_config.items(section):         
            print_format[section].append(key)
            config_attribs.append(key)
            data_dict[key] = parse_value(value) if key not in NO_PARSE else value
            
    data_dict['save_root'] = Path(data_dict['save_root']) / (data_dict['name'] + '_' + data_dict['mode'] + f'_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}')
    if data_dict['mode'] == 'train':
        os.system(f"mkdir {data_dict['save_root']}")
    Config = namedtuple('Config', config_attribs)
    cfg = Config(**data_dict)
    return cfg, print_format


def parse_value(value):
    if value.replace('.', '', 1).replace('+', '', 1).replace('-', '', 1).replace('e', '', 1).isdigit():
        return literal_eval(value)
    elif value == 'True' or value == 'False':
        if value == 'True':
            return True
        else:
            return False
    elif value == 'None':
        return None
    elif ',' in value:
        is_number = any(char.isdigit() for char in value.split(',')[0])
        items_list = value.split(',')

        if '' in items_list:
            items_list.remove('')
        if is_number:
            return [literal_eval(val) for val in items_list]
        else:
            if '\"' in items_list[0] and '\'' in items_list[0]:
                return [literal_eval(val.strip()) for val in items_list]
            else:
                return [val.strip() for val in items_list]
    else:
        return value


def override_cfg(override, cfg):
    equality_split = override.split('=')
    num_equality = len(equality_split)
    assert num_equality > 0
    if num_equality == 2:
        override_dict = {equality_split[0]: parse_value(equality_split[1])}
    else:
        keys = [equality_split[0]]          # First key
        keys += [equality.split(',')[-1].strip() for equality in equality_split[1:-1]]  # Other keys
        values = [equality.replace(', ' + key, '') for equality, key in zip(equality_split[1:-1], keys[1:])]  # Get values other than last field
        values.append(equality_split[-1])   # Get last value
        values = [value.replace('[', '').replace(']', '') for value in values]

        override_dict = {key: parse_value(value) for key, value in zip(keys, values)}

    cfg_dict = cfg._asdict()            
    Config = namedtuple('Config', tuple(set(cfg._fields + tuple(override_dict.keys()))))
    cfg_dict.update(override_dict) 
    cfg = Config(**cfg_dict)

    return cfg


def print_cfg(cfg, print_format):
    cfg_dict = cfg._asdict()
    for section in print_format.keys():
        print('[%s]' % section)
        for item in print_format[section]:
            print('%s=%s' % (item, cfg_dict[item]))
        print('')


def save_cfg(cfg, print_format):
    cfg_dict = cfg._asdict()
    out_config = configparser.ConfigParser()
    for section in print_format.keys():
        out_config.add_section(section)
        for key in print_format[section]:
            out_config[section][key] = str(cfg_dict[key])
    
    with open( cfg.save_root / 'config.ini', 'w') as configfile:
        out_config.write(configfile)








