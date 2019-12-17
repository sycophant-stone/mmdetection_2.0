

class PerfStatistic(object):
    def __init__(self):
        # layer: FLOPs
        self.layer_FLOPs_map = {}

        # layer: mem costage.
        self.layer_mem_map = {}

        # layer: info
        self.layer_info_map = {}

        # default padding value.
        self.DEFAULT_PADDING_VALUE = 0

        # first flag
        self.flag_first_call = False

        #
        self.which_net=""
    def set_net_name(self, net_name):
        '''
        set net name
        :param net_name:
        :return:
        '''
        self.which_net=net_name

    def set_layer_info(self, key, value):
        '''
        set layer info
        :param key:
        :param value:
        :return:
        '''
        self.layer_info_map[key] = value

    def get_layer_info_by_key(self,key):
        return self.layer_info_map[key]

    def set_layer_flops(self, key, value):
        '''
        set layer flop
        :param key:
        :param value:
        :return:
        '''
        self.layer_FLOPs_map[key] = value

    def get_layer_flops_by_key(self,key):
        return self.layer_FLOPs_map[key]


    def set_layer_mem(self, key, value):
        '''
        set layer mem
        :param key:
        :param value:
        :return:
        '''
        self.layer_mem_map[key] = value

    def get_layer_mem_by_key(self, key):
        '''
        get layer mem costage by key.
        :param key:
        :return:
        '''
        return self.layer_mem_map[key]

    def update_statis(self, name, input_shape):
        '''
        calculation the mem-costage and FLOPs...
        :return:
        '''
        layer_info = self.get_layer_info_by_key(name)
        input_chn = layer_info['ic']
        output_chn = layer_info['oc']
        kernel_size = layer_info['k']
        padding_size = layer_info['p']
        stride_size = layer_info['s']
        layer_info['ih'] = input_shape[0]
        layer_info['iw'] = input_shape[1]
        mem_costage = kernel_size*kernel_size*input_chn*output_chn
        print("%s's  input_shape: %s"%(name,input_shape))
        in_h,in_w = input_shape
        out_h = (in_h-kernel_size+2*padding_size)/stride_size + 1
        out_w = (in_w-kernel_size+2*padding_size)/stride_size + 1
        layer_info['oh'] = out_h
        layer_info['ow'] = out_w
        flops = kernel_size*kernel_size*out_h*out_w*input_chn*output_chn
        self.layer_FLOPs_map[name] = flops
        self.layer_mem_map[name] = mem_costage

    def evaluation_result(self):
        # print("==============================================")
        # print("evaluation [[ %s  ]]  results: "%(self.which_net))
        # print("==============================================")
        namelist = self.layer_mem_map.keys()
        all_mem=0
        all_flops=0
        for name in namelist:
            layer_info = self.get_layer_info_by_key(name)
            print("[  %s  ]    input,ouput format is [b,c,h,w]")
            print("[  %s  ]    input[b,c,h,w]: (,%s,%s,%s)"%(name, layer_info['ic'], layer_info['ih'], layer_info['iw']))
            print("[  %s  ]    output[b,c,h,w]: (,%s,%s,%s)" % (name, layer_info['oc'], layer_info['oh'], layer_info['ow']))
            print("[  %s  ]    mem costage:   %s"%(name, self.layer_mem_map[name]))
            print("[  %s  ]    flops      :   %s" % (name, self.layer_FLOPs_map[name]))
            print("---------------------------------------------------------")
            all_mem = all_mem + self.layer_mem_map[name]
            all_flops = all_flops + self.layer_FLOPs_map[name]
        # if want to show the resnet params..
        # print("[  All  ]    mem costage:   %s" % (all_mem))
        # print("[  All  ]    flops      :   %s" % (all_flops))
        # print("---------------------------------------------------------")


