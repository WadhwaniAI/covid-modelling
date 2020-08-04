import yaml


def read_config(path, backtesting=False):
    default_path = os.path.join(os.path.dirname(path), 'default.yaml')
    with open(default_path) as default:
        config = yaml.load(default, Loader=yaml.SafeLoader)
    with open(path) as configfile:
        new = yaml.load(configfile, Loader=yaml.SafeLoader)
    for k in config.keys():
        if type(config[k]) is dict and new.get(k) is not None:
            config[k].update(new[k])
    model_params = config['model_params']
    if backtesting:
        config['base'].update(config['backtesting'])
    else:
        config['base'].update(config['run'])
    config = config['base']
    return config, model_params
