import yaml

# "evaluation_config/eva_config.yaml"
class YamlReader(object):
    def __init__(self, src_yaml_file):
        self.yaml_src_file = src_yaml_file
        with open(src_yaml_file) as f:
            self.yaml_info = yaml.load(f)


    def get_yaml_by_keyname(self, key_class1, key_class2):
        '''
        get yaml value by keyname
        :param key_class1:
        :param key_class2:
        :return:
        '''
        try:
            return self.yaml_info[key_class1][key_class2]
        except:
            raise Exception("%s %s not exist in %s"%(key_class1,key_class2,self.yaml_src_file))

