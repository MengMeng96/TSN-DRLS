import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("../")
sys.path.append("../..")
sys.path.append(".")
from src.Environment import *
from src.param import *
from src.pytorch_op import *
from src.pytorch_gcn import *
import time
from src.Agent import *
from src.Data.RingSpecific import *


def test():
    actor_agent = Actor_Agent(args.policy_input_dim, args.hid_dims, args.output_dim, args.max_depth)
    data_gene = LadderGenerater()
    node_num = 14
    print("node number", node_num)
    data_gene.gene_all(node_num=node_num, eps=0.35, rand_min=5, rand_max=10, tt_num=60000,
                       delay_min=64, delay_max=512, pkt_min=72, pkt_max=1526, hop=1, dynamic=True)
    actor_agent.env = Environment(data_gene)  # DataGenerater(node_num))
    actor_agent.init()
    actor_agent.env.enforce_next_query()

    manual = False
    policy_inputs, policy_outputs, edge, time_slot, edge_selected_mask, cycle, LD_score, flag = actor_agent.invoke_model(
        manual)
    print(actor_agent.env.tt_query_id)
    reward, done, reason = actor_agent.env.step(edge, time_slot, LD_score)
    criterion = policy_loss(edge_selected_mask)
    optimizer = torch.optim.Adam(Actor_Agent.parameters(actor_agent), lr=0.001)
    edge_loss = criterion(policy_outputs, reward)
    optimizer.zero_grad()
    edge_loss.backward()
    optimizer.step()


def schedule_single_stream(actor_agent, environment):
    done = 0
    manual = True
    edge_path = []
    while done == 0:
        valid_edges, policy_inputs, time_inputs, cycle, time_offset, flow_length, max_delay = environment.translate_data_to_inputs()
        input = torch.FloatTensor(policy_inputs)
        policy_output = actor_agent.forward(input, environment.graph.reachable_edge_matrix)

        scope = range(len(policy_inputs))
        edge_candidate = []
        if manual:
            scope = valid_edges
        if len(scope) == 0:
            return -1, []
        for edge_id in scope:
            edge_candidate.append([edge_id, policy_output[0, edge_id]])
        edge_candidate = sorted(edge_candidate, key=lambda edge_info: edge_info[1])
        # print(scope)
        # print("edge network: ", edge_candidate)
        # 选边
        edge_info = edge_candidate[-1]
        edge = environment.graph.edges[edge_info[0]]
        # print("edge network: ", [it[0] for it in edge_candidate], edge.id)
        # TODO 记录路由之后整体找时隙
        edge_path.append(edge)

        if edge.end_node.is_destination_node:
            done = 1
        else:
            # 中间步骤，维护更新节点和边信息
            for node in environment.graph.nodes.values():
                node.is_source_node = 0
            edge.end_node.is_source_node = 1
            # print(edge.start_node.id, edge.end_node.id)
            for edge in environment.graph.edges.values():
                edge.refresh()
    if not environment.enforce_next_query():
        done = -2
    return done, edge_path


def bunchCalculate(actor_agent, environment, node_info=None, node_mat=None, tt_flow_requirements=None):
    # Schedule
    continue_to_schedule = True
    start_time = time.time()
    cur_time = time.time()
    while continue_to_schedule:
        # 找到路径
        done, edge_path = schedule_single_stream(actor_agent, environment)
        if done == -1:
            print("TT_flow", environment.cur_tt_flow_id, "failed")
            break
        # 找到时隙
        delay, route = environment.graph.select_time_slot_no_waiting(edge_path, environment.tt_flow_cycle)
        # 转化为GCL表项
        environment.TSN_schedule.update(environment.cur_tt_flow_id, route)
        # 维护网络资源信息
        for [edge, queueID, receive_time_slot, send_time_slot] in route:
            # print("  ", edge.id, queueID, receive_time_slot, send_time_slot)
            edge.occupy_queue_and_time_slot(queueID, receive_time_slot, send_time_slot, environment.tt_flow_cycle)

        print("TT_flow", environment.cur_tt_flow_id,
              "cycle", environment.tt_flow_cycle,
              "usage", environment.edge_usage(),
              "use time", time.time() - cur_time,
              "delay", delay)
        if done == -2:
            print("Schedule finished")
            break
    total_time = time.time() - start_time
    average_time = total_time / environment.cur_tt_flow_id
    usage = environment.edge_usage()

    print(total_time, average_time, usage)
    return environment


def getBunchCalculateResult(environment):
    return environment.schedule.sche


def singleCalculate(environment, query):
    actor_agent = Actor_Agent()
    environment.enforce_specific_query(query)
    information = {}
    information["done"] = -1
    information["delay"] = -1
    information["reward"] = -1
    reward = -1
    done = 0
    manual = True
    while done == 0:
        valid_edges, policy_inputs, time_inputs, cycle, time_offset, flow_length, max_delay = environment.translate_data_to_inputs()
        input = torch.FloatTensor(policy_inputs)
        policy_output = actor_agent.forward(input, environment.graph.reachable_edge_matrix)

        scope = range(len(policy_inputs))
        edge_candidate = []
        if manual:
            scope = valid_edges
            if len(valid_edges) == 0:
                return actor_agent, environment, information
        for edge_id in scope:
            edge_candidate.append([edge_id, policy_output[0, edge_id]])
            edge_candidate = sorted(edge_candidate, key=lambda edge_info: edge_info[1])
        # print("edge network: ", edge_candidate)
        # 选边
        edge_info = edge_candidate[-1]
        edge = environment.graph.edges[edge_info[0]]

        # 时延应该 从发出时间算，而不是从0时刻开始算
        start_time = -1
        offset = -1
        if len(environment.tt_flow_time_record) > 1:
            start_time = environment.tt_flow_time_record[-1] - environment.tt_flow_time_record[1]
            offset = environment.tt_flow_time_record[-1] % args.global_cycle
        time_slot, LD_score = edge.find_time_slot(start_time, offset, cycle, flow_length, max_delay)
        reward, done, reason = environment.step(edge, time_slot, LD_score)
        # print(edge.id, time_slot, done, scope)

    delay = -1
    if done == 1:
        delay = environment.tt_flow_time_record[-1] - environment.tt_flow_time_record[1]
    # print("cumulated reward", cumulated_reward, "done", done)

    information["done"] = done
    information["delay"] = delay
    information["reward"] = reward
    done, delay, reward
    return environment, information


def getSingleCalculateResult(environment):
    return environment.current_stream_schedule.sche


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


