import yaml
import utils.shell as shell
import utils.json_encoder as json_encoder
import utils.filesystem as filesystem
from jinja2 import Template
import utils.concat as yaml_tags_concat

from research.rlogger import research_logger_info as rlogger

def config_to_template_vars(step, config_file):
    store_config = yaml.load(filesystem.file_get_contents(config_file), Loader=yaml.Loader)
    template_vars = {}
    for var in store_config[step]:
        template_vars[var] = store_config[step][var]
    return template_vars


def fcos_run():
    rlogger("fcos pipeline")
    

def Test_fcos_run():
    fcos_run()

if __name__=='__main__':
    Test_fcos_run()
