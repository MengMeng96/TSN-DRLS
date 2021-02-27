import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append(".")
from Environment import *
from param import *
from Data.Random import *
from pytorch_op import *
from pytorch_gcn import *
import time
from FlaskServer import *
from Agent import *


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


def schedule_single_stream(actor_agent, environment, total_exps_cnt, max_exps_number, success_exps, fail_exps, network_index):
    environment.enforce_next_query()
    experiences = []
    done = 0
    manual = False
    while done == 0:
        valid_edges, policy_inputs, time_inputs, cycle, time_offset, flow_length, max_delay = environment.translate_data_to_inputs()
        input = torch.FloatTensor(policy_inputs)
        policy_output = actor_agent.forward(input, environment.graph.reachable_edge_matrix)

        scope = range(len(policy_inputs))
        edge_candidate = []
        if manual:
            scope = valid_edges
        for edge_id in scope:
            edge_candidate.append([edge_id, policy_output[0, edge_id]])
        edge_candidate = sorted(edge_candidate, key=lambda edge_info: edge_info[1])
        # print("edge network: ", edge_candidate)
        # 选边
        edge_info = edge_candidate[-1]
        edge = environment.graph.edges[edge_info[0]]
        # print("edge network: ", [it[0] for it in edge_candidate], edge.id)
        # 计算策略网络梯度梯度：给出选中的边的mask
        one_hot_mask = np.zeros([1, len(environment.graph.edges)])
        one_hot_mask[0, edge.id] = 1
        # print(one_hot_mask)

        # 时延应该 从发出时间算，而不是从0时刻开始算
        start_time = -1
        offset = -1
        if len(environment.tt_flow_time_record) > 1:
            start_time = environment.tt_flow_time_record[-1] - environment.tt_flow_time_record[1]
            offset = environment.tt_flow_time_record[-1] % args.global_cycle
        time_slot, LD_score = edge.find_time_slot(start_time, offset, cycle, flow_length, max_delay)

        reward, done, reason = environment.step(edge, time_slot, LD_score)
        # print(edge.id, time_slot)

        if time_slot >= 0:
            exp = {}
            exp["input"] = policy_inputs
            exp["output"] = policy_output
            exp["one_hot_mask"] = one_hot_mask
            exp["reward"] = reward
            exp["netwrok_id"] = network_index
            exp["reachable_edge_matrix"] = environment.graph.reachable_edge_matrix
            experiences.append(exp)
        elif len(experiences) > 0:
            experiences[-1]["reward"] = reward
    cumulated_reward = [exp["reward"] for exp in experiences]
    cumulated_reward = discount(cumulated_reward, 0.8)
    # print(cumulated_reward)
    for i in range(len(experiences)):
        experiences[-i - 1]["reward"] = cumulated_reward[-i - 1]
    delay = -1
    reward = -1
    if done == 1:
        for exp in experiences:
            total_exps_cnt += 1
            if len(success_exps) < max_exps_number:
                success_exps.append(exp)
            else:
                k = random.randint(0, total_exps_cnt - 1)
                if k < max_exps_number:
                    success_exps[k] = exp
        delay = environment.tt_flow_time_record[-1] - environment.tt_flow_time_record[1]
        reward = experiences[-1]["reward"]
    else:  # if reason == "Visited edge or Not adjacent edge":
        print(reason, environment.tt_query_id, environment.cur_tt_flow_id, done)
        fail_exps.extend(experiences)
    # print("cumulated reward", cumulated_reward, "done", done)
    return done, delay, reward


def schedule(actor_agent, environment, network_index, flow_count_record, total_exps_cnt, max_exps_number, success_exps, fail_exps):
    # Schedule
    failed_cnt = 0
    continue_to_schedule = True
    start_time = time.time()
    cur_time = time.time()
    while continue_to_schedule:
        done, delay, reward = schedule_single_stream(actor_agent, environment, total_exps_cnt, max_exps_number, success_exps, fail_exps, network_index)
        if done == -1:
            failed_cnt += 1
            if failed_cnt >= 10:
                continue_to_schedule = False
        # else:
        #     print("TT_flow", environment.cur_tt_flow_id,
        #           "cycle", environment.tt_flow_cycle,
        #           "usage", environment.edge_usage(),
        #           "use time", time.time() - cur_time,
        #           "delay", delay,
        #           "reward", reward)

        if environment.cur_tt_flow_id == 59999:
            continue_to_schedule = False

    # Update
    edge_usage = environment.edge_usage()
    print(environment.cur_tt_flow_id, edge_usage, len(success_exps), len(fail_exps))
    flow_count_record[network_index] = 1 - edge_usage


def update(actor_agent, success_exps, fail_exps, flow_count_record, learning_rate):
    cur_exps = random.sample(success_exps, min(len(success_exps), 800))
    cur_exps.extend(random.sample(fail_exps, min(len(fail_exps), 200)))
    criterion = policy_loss()
    optimizer = torch.optim.Adam(Actor_Agent.parameters(actor_agent), lr=learning_rate)
    optimizer.zero_grad()
    for exp in cur_exps:
        policy_input = exp["input"]
        input = torch.FloatTensor(policy_input)
        reachable_edge_matrix = exp["reachable_edge_matrix"]
        # show_parameters(actor_agent.output_layer)
        policy_output = actor_agent.forward(input, reachable_edge_matrix)
        edge_selected_mask = exp["one_hot_mask"]
        value_reward = exp["reward"]
        netwrok_id = exp["netwrok_id"]
        if value_reward > 0:
            value_reward = value_reward * (1 + flow_count_record[netwrok_id] - 0.5)

        edge_loss = criterion(policy_output, value_reward, edge_selected_mask)

        edge_loss.backward()
        # print([x.grad for x in optimizer.param_groups[0]['params']])
        # print("*****************************************")
        # for i in actor_agent.output_layer.named_parameters():
        #     print(i)
    optimizer.step()
        # for i in actor_agent.output_layer.named_parameters():
        #     print(i)


def show_parameters(layer):
    for item in list(layer.named_parameters()):
        print(item)


def main():
    # System init
    np.random.seed()
    total_exps_cnt = 0
    max_exps_number = 20000
    success_exps = []
    fail_exps = []
    network_index = 0
    flow_count_record = {}

    # Agent init
    actor_agent = Actor_Agent(args.policy_input_dim, args.hid_dims, args.output_dim, args.max_depth)
    actor_agent.load_state_dict(torch.load('../models/Random_Model/temp/params.pkl'))
    data_gene = DataGenerater()
    node_num = 15
    min_usage = 0.5
    last_update = [-1, -1]
    learning_rate = 0.00001 # 3.433683820292515e-07

    dataset = f"../jhy/Random_NetWork/n5/0/"
    # 创建结果文件
    # write_result_init()
    while True:
        # args.data_path = dataset
        data_gene.gene_all(node_num=node_num, eps=0.35, rand_min=5, rand_max=10, tt_num=60000,
                           delay_min=64, delay_max=512, pkt_min=72, pkt_max=1526, hop=1, dynamic=True)
        network_index += 1
        environment = Environment(data_gene)  # DataGenerater(node_num))

        # Schedule
        schedule(actor_agent, environment, network_index, flow_count_record, total_exps_cnt, max_exps_number, success_exps, fail_exps)
        usage = environment.edge_usage()
        flow_count_record[network_index] = usage
        if usage < min_usage or usage < 0.5:
            torch.save(actor_agent.state_dict(), '../models/Random_Model/temp/params.pkl')
            last_update = [network_index, usage]
            min_usage = usage
            learning_rate *= 0.9

        # Update

        update(actor_agent, success_exps, fail_exps, flow_count_record, learning_rate)
        print(network_index, last_update, learning_rate)
        # show_parameters(actor_agent.gcn_layers.prepare_layer3)

        # Result
        # print(environment.schedule.toString())


if __name__ == '__main__':
    main()


