import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("../")
sys.path.append("./")
sys.path.append("./TorchVersion")
from src.Environment import *
from src.param import *
import time
from DRLS import *


if __name__ == '__main__':
    # System init
    np.random.seed()
    output_directory = args.output_directory
    input_file = args.data_path

    # Agent init
    actor_agent = Actor_Agent()
    actor_agent.load_state_dict(torch.load('./models/Random_Model/temp/params.pkl'))
    tsn_info = json.load(open(input_file, encoding='utf-8'))
    environment = Environment(tsn_info)
    environment = bunchCalculate(actor_agent, environment)
    environment.TSN_schedule.show()
    environment.TSN_schedule.write_result(output_directory)

    # data_gene = RingSpecificGenerater()
    # if not os.path.exists(f'../resource/PCL_NetWork/test'):
    #     os.mkdir(f'../resource/PCL_NetWork/test')
    # for i in range(1, 2):
    #     node_num = 10
    #     data_gene.gene_all(node_num=node_num, eps=0.35, rand_min=5, rand_max=10, tt_num=8,
    #                        delay_min=64, delay_max=512, pkt_min=72, pkt_max=1526, hop=1, dynamic=True)
    #     data_gene.transform_schedule_to_node_info(environment.TSN_schedule.result)
    #     data_gene.write_to_file(filename=f"PCL_NetWork/test/{i}")