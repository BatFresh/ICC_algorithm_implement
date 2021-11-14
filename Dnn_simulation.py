import math
from algorithm import heft_time_reservation,heft,greedy,lbck,greedy_time_reservation,get_algorithm_timelist,set_paramerters,greedy_nlook_back,greedy_time_reservation_nlook_back,heft_n_look_back,NSGA_n_look_back
from Dataset import get_connect_task_graph,generate_randomtimeline,get_connect_multiple_task_graph,default_timewindow,lookahead_window_size
import copy
# 'Global View Greedy-Reservation','Global View HEFT','Global View Greedy',"NSGA",
# "Partial View Greedy-Reservation-Improved",'Partial View HEFT-Improved','Partial View Greedy-Improved'
ALGOR_NAME_LIST = [
"Partial View Greedy-Reservation",'Partial View HEFT','Partial View Greedy',"Partial View NSGA"
]
# 

# 'Global View LBCK'
# DNN FLOP:0 Google :1
defalut_tasktype = [0,0,0]

defalut_inter_num = 1

defalut_edge_nums = 4
defalut_max_edge_nums = 10

defalut_delta = 0.01

defalut_avaratio = 0.75
defalut_sigma = 0.05

defalut_start_ratio = 0.2
defalut_start_sigma = 0.05

# compare nice_expdata1
# defalut_start_ratio = 0.01
# defalut_start_sigma = 0.1

defalut_max_cpu_capbility = 305

# default_timewindow = 0.02
# default_timewindow_number = 50
# default_totaltime = default_timewindow * default_timewindow_number

# defalut_deadline = 50
defalut_request_number = [200,200,200]

# --------------指标---------------
# 获取算法结果的完成时间
def get_max_time(anslist):
    max_time = -1
    if anslist == 2 or anslist == 3:
        return max_time
    for tmp in anslist:
        tmp.sort(key=lambda x: x[2], reverse=True)

        if len(tmp) != 0:
            max_time = max(max_time, tmp[0][1])

    return max_time

# 获取算法结果的吞吐率
def get_throught_ratio(anslist,deadline):
    # 2 -> budget 无法调度
    # 3 -> 有效时隙无法调度
    max_time = 0
    if anslist == 3 or anslist == 2:
        return 0
    for tmp in anslist:
        tmp.sort(key=lambda x: x[2], reverse=True)
        if len(tmp) != 0:
            max_time = max(max_time, tmp[0][2])
    if deadline > max_time:
        return 1
    else:
        return 0
    # return max_time

# 获取算法运行时间
def get_run_time():
    import time

    # starttime = time.clock()
    # heft_time_reservation()
    # endtime = time.clock()
    # heft_time_reservation_time = endtime - starttime

    # starttime = time.clock()
    # greedy_time_reservation()
    # endtime = time.clock()
    # greedy_time_reservation_time = endtime - starttime

    # starttime = time.clock()
    # heft()
    # endtime = time.clock()
    # heft_time = endtime - starttime

    # starttime = time.clock()
    # lbck()
    # endtime = time.clock()
    # lbck_time = endtime - starttime

    # starttime = time.clock()
    # greedy()
    # endtime = time.clock()
    # greedy_time = endtime - starttime


    starttime = time.clock()
    greedy_time_reservation_nlook_back(lookahead_window_size)
    endtime = time.clock()
    greedy_time_reservation_nlook_back_time = endtime - starttime

    starttime = time.clock()
    heft_n_look_back(lookahead_window_size)
    endtime = time.clock()
    heft_n_look_back_time = endtime - starttime

    starttime = time.clock()
    greedy_nlook_back(lookahead_window_size)
    endtime = time.clock()
    greedy_nlook_back_time = endtime - starttime

    starttime = time.clock()
    NSGA_n_look_back(lookahead_window_size)
    endtime = time.clock()
    NSGA_nlook_back_time = endtime - starttime
    
    return [greedy_time_reservation_nlook_back_time,heft_n_look_back_time,greedy_nlook_back_time,NSGA_nlook_back_time]

def result_ratiocal(result_dict_list,inter_num):

    avg_dict = {}
    rangelen = len(result_dict_list[0][ALGOR_NAME_LIST[0]])
    for i in range(len(ALGOR_NAME_LIST)):
        avg_dict[ALGOR_NAME_LIST[i]] = [0 for j in range(rangelen)]
    
    # avg_time_dict = time_dict_list[0]
    for i in range(len(result_dict_list)):
        for key in avg_dict.keys():
            for j in range(len(avg_dict[key])):
                # 遍历参数组
                avg_dict[key][j] += result_dict_list[i][key][j][0]

    for key in avg_dict.keys():
        for j in range(len(avg_dict[key])):
            avg_dict[key][j] /= inter_num
    return avg_dict

def result_timecal(result_dict_list,inter_num):
    avg_dict = {}
    tmp_dict = {}

    rangelen = len(result_dict_list[0][ALGOR_NAME_LIST[0]])
    for i in range(len(ALGOR_NAME_LIST)):
        tmp_dict[ALGOR_NAME_LIST[i]] = [[] for j in range(rangelen)]

    # avg_time_dict = time_dict_list[0]
    for i in range(len(result_dict_list)):
        for key in tmp_dict.keys():
            for j in range(len(result_dict_list[i][key])):
                # 遍历参数组
                if result_dict_list[i][key][j][0] != -1:
                    tmp_dict[key][j].append(result_dict_list[i][key][j][0])

    for i in range(len(ALGOR_NAME_LIST)):
        avg_dict[ALGOR_NAME_LIST[i]] = [0 for j in range(len(tmp_dict[ALGOR_NAME_LIST[i]]))]

    for i in range(len(ALGOR_NAME_LIST)):
        for j in range(len(tmp_dict[ALGOR_NAME_LIST[i]])):
            avg_dict[ALGOR_NAME_LIST[i]][j] = sum(tmp_dict[ALGOR_NAME_LIST[i]][j])

    for key in avg_dict.keys():
        for j in range(len(avg_dict[key])):
            if len(tmp_dict[key][j]) != 0:
                avg_dict[key][j] /= len(tmp_dict[key][j])
            else:
                avg_dict[key][j] = -1
    return avg_dict
 
# --------------指标---------------

def taskgraph_exp(data_prefix, taskgraph,**kwargs):
    # from code02 import set_paramerters,get_time_list
    import pandas as pd

    avatimelist = []
    avatime_ratio = defalut_avaratio
    edge_computer_cability = []
    resouce_upbound = []
    time_list = [[] for i in  range(len(ALGOR_NAME_LIST))]
    ratio_list = [[] for i in  range(len(ALGOR_NAME_LIST))]

    request_number = defalut_request_number

    # n_look = 30
    # file_prefix = 'exp1_edge_num_change'
    max_edge_num = defalut_max_edge_nums
    edge_nums = defalut_edge_nums

    max_cpu_capbility = defalut_max_cpu_capbility
    delta = defalut_delta
    mu = defalut_avaratio
    ratio_sigma = defalut_sigma

    window_size = default_timewindow
    start_ratio = defalut_start_ratio
    start_sigma = defalut_start_sigma

    change_edge_num = True

    if "ava_ratio" in kwargs:
        avatime_ratio = kwargs['ava_ratio']


    if 'max_edge_num' in kwargs:
        max_edge_num = kwargs['max_edge_num']

    if 'change_edge_num' in kwargs:
        change_edge_num = kwargs['change_edge_num']

    if 'max_cpu_capbility' in kwargs:
        max_cpu_capbility = kwargs['max_cpu_capbility']

    if 'decision_time_list' in kwargs:
        decision_time_list = kwargs['decision_time_list']

    if 'delta' in kwargs:
        delta = kwargs['delta']

    if 'mu' in kwargs:
        mu = kwargs['mu']

    if 'ratio_sigma' in kwargs:
        ratio_sigma = kwargs['ratio_sigma']

    if 'request_number' in kwargs:
        request_number = kwargs['request_number']

    if 'start_ratio' in kwargs:
        start_ratio = kwargs['start_ratio']

    if 'start_sigma' in kwargs:
        start_sigma = kwargs['start_sigma']

    if 'window_size' in kwargs:
        window_size = kwargs['window_size']

    task_info = None

    if "task_info" in kwargs:
        task_info = kwargs['task_info']

    pre,succ,workload,datasize,taskindex2order_map,order2taskindex_map,order2subtaskindex_map = task_info

    # set_paramerters()
    if change_edge_num:
        # edge_num = 3
        decision_time_list = []
        avatimelist = []

        new_decision_time_list,new_avatimelist = generate_randomtimeline(num_edges=max_edge_num,
        start_ratio=start_ratio,start_sigma=start_sigma,ava_ratio=avatime_ratio,ratio_sigma=ratio_sigma)
        for edge_num in range(3, max_edge_num):
            edge_num_time_list = []

            # reset ava_time_list
            decision_time_list = copy.deepcopy(new_decision_time_list[:edge_num])
            avatimelist = copy.deepcopy(new_avatimelist[:edge_num])

            # reset random time
            random_time = [[delta for i in range(len(workload))] for i in range(edge_num)]
            # reset W
            W = [[12.5 for i in range(edge_num)] for i in range(edge_num)]

            # reset edge_computer_capblity
            edge_computer_cability = [max_cpu_capbility for i in range(edge_num)]

            # reset resouce upbound
            resouce_upbound = []

            for tmpava_bydevice in avatimelist:
                tmpsum = 0
                for tmpinterval in tmpava_bydevice:
                    tmplen = tmpinterval[1] - tmpinterval[0]
                    tmpsum = tmpsum + tmplen
                resouce_upbound.append(tmpsum)

            set_paramerters(workload=workload, datasize=datasize, pre=pre, succ=succ, num_edges=edge_num, ava_time_list=avatimelist, random_time=random_time, bandwidth_edge=W,
                            taskindex2order_map=taskindex2order_map,order2taskindex_map=order2taskindex_map,order2subtaskindex_map=order2subtaskindex_map,window_size=window_size,
                            edge_computer_capability=edge_computer_cability, resouce_upbound=resouce_upbound,decision_time_list=decision_time_list)


            edge_num_time_list += get_algorithm_timelist()

            # 将配置信息写入文件当中
            # with open("{0}_{1}_paramerters.txt".format(data_prefix, edge_num), 'a+') as file:
            #     file.write("\n")
            #     file.write("edge num: {0}\n avatime_list: {1}\n avatime_raido: {2}\n edge computer capability: {3}\n "
            #                "resource upbound: {4}\n n look: {5} random time: {6}\n bandwidth edge: {7} resouce upbound: {8}\n".format(
            #         edge_num, avatimelist, avatime_radio, edge_computer_cability, resouce_upbound, n_look, random_time, W, resouce_upbound
            #     ))

            # 实验数据记录
            for i in range(len(edge_num_time_list)):
                # ratio_list[i].append([get_throught_ratio(edge_num_time_list[i],deadline=defalut_deadline)])
                time_list[i].append([get_max_time(edge_num_time_list[i])])


    else:
        edge_num = edge_nums

        edge_num_time_list = []

        # reset ava_time_list
        if 'avatimelist' in kwargs:
            avatimelist = kwargs['avatimelist']

        if 'decision_time_list' in kwargs:
            decision_time_list = kwargs['decision_time_list']
        # else:
        #     avatimelist= [generate_ava_time_and_unava_time(avatime_radio, 20, 300) for i in range(edge_num)]

        # reset random time
        random_time = [[delta for i in range(len(workload))] for i in range(edge_num)]
        # reset W
        W = [[12.5 for i in range(edge_num)] for i in range(edge_num)]

        # reset edge_computer_capblity
        edge_computer_cability = [max_cpu_capbility for i in range(edge_num)]

        # reset resouce upbound
        resouce_upbound = []

        for tmpava_bydevice in avatimelist:
            tmpsum = 0
            for tmpinterval in tmpava_bydevice:
                tmplen = tmpinterval[1] - tmpinterval[0]
                tmpsum = tmpsum + tmplen
            resouce_upbound.append(tmpsum)

        set_paramerters(workload=workload, datasize=datasize, pre=pre, succ=succ, num_edges=edge_num,window_size=window_size,
                        ava_time_list=avatimelist, random_time=random_time, bandwidth_edge=W,decision_time_list=decision_time_list,
                        taskindex2order_map=taskindex2order_map,order2taskindex_map=order2taskindex_map,order2subtaskindex_map=order2subtaskindex_map,
                        edge_computer_capability=edge_computer_cability, resouce_upbound=resouce_upbound)

        # tmptimelist = get_time_list()

        edge_num_time_list += get_algorithm_timelist()

        # 将配置信息写入文件当中
        # with open("{0}_{1}_paramerters.txt".format(file_prefix, edge_num), 'a+') as file:
        #     file.write("\n")
        #     file.write("edge num: {0}\n avatime_list: {1}\n avatime_raido: {2}\n edge computer capability: {3}\n "
        #                "resource upbound: {4}\n n look: {5} random time: {6}\n bandwidth edge: {7} resouce upbound: {8}\n".format(
        #         edge_num, avatimelist, avatime_radio, edge_computer_cability, resouce_upbound, n_look, random_time, W,
        #         resouce_upbound
        #     ))

        for i in range(len(edge_num_time_list)):
            # 实验数据记录
            # ratio_list[i].append(get_throught_ratio(edge_num_time_list[i],deadline=defalut_deadline))
            time_list[i].append(get_max_time(edge_num_time_list[i]))

    # 实验数据分算法记录
    time_dict = {}
    for i in range(len(time_list)):
        time_dict[ALGOR_NAME_LIST[i]] = time_list[i]

    # ratio_dict = {}
    # for i in range(len(ratio_list)):
    #     ratio_dict[ALGOR_NAME_LIST[i]] = ratio_list[i]

    return time_dict

def taskgraph_exp_runtime(data_prefix, taskgraph,**kwargs):
    '''
    实验7
    获取不同边缘服务器数量下两个算法的任务运行时间
    需要修改所有边缘服务器相关参数：
    * edge_num
    * ava_time_list
    * random_time
    * W
    * edge_computer_capbility
    * resource_upbound
    :return:
    '''
    # from code02 import set_paramerters,get_time_list
    import pandas as pd

    avatimelist = []

    avatime_ratio = defalut_avaratio
    sigma = defalut_sigma

    edge_computer_cability = []
    resouce_upbound = []
    runtime_list = [[] for i in  range(len(ALGOR_NAME_LIST))]
    
    max_edge_num = defalut_max_edge_nums
    edge_nums = defalut_edge_nums

    max_cpu_capbility = defalut_max_cpu_capbility
    delta = defalut_delta

    window_size = default_timewindow
    start_ratio = defalut_start_ratio
    start_sigma= defalut_start_sigma

    request_number = defalut_request_number
    change_edge_num = True

    # set big task graph paramerters
    pre,succ,workload,datasize,taskindex2order_map,order2taskindex_map,order2subtaskindex_map = get_connect_multiple_task_graph(request_number,taskgraph,tasktype=defalut_tasktype)

    # if 'n_look' in kwargs:
    #     n_look = kwargs['n_look']

    if 'max_edge_num' in kwargs:
        max_edge_num = kwargs['max_edge_num']

    if 'change_edge_num' in kwargs:
        change_edge_num = kwargs['change_edge_num']

    if 'max_cpu_capbility' in kwargs:
        max_cpu_capbility = kwargs['max_cpu_capbility']

    if 'decision_time_list' in kwargs:
        decision_time_list = kwargs['decision_time_list']

    if 'delta' in kwargs:
        delta = kwargs['delta']

    if 'sigma' in kwargs:
        sigma = kwargs['sigma']

    if 'window_size' in kwargs:
        window_size = kwargs['window_size']
    task_info = None

    if "task_info" in kwargs:
        task_info = kwargs['task_info']
        
    pre,succ,workload,datasize,taskindex2order_map,order2taskindex_map,order2subtaskindex_map = task_info

    # set_paramerters()
    # edge_num = 3
    decision_time_list = []
    avatimelist = []

    new_decision_time_list,new_avatimelist = generate_randomtimeline(num_edges=max_edge_num,
        start_ratio=start_ratio,start_sigma=start_sigma,ava_ratio=avatime_ratio,ratio_sigma=sigma)

    for edge_num in range(3, max_edge_num):
        edge_num_time_list = []

        # reset ava_time_list
        decision_time_list = copy.deepcopy(new_decision_time_list[:edge_num])
        avatimelist = copy.deepcopy(new_avatimelist[:edge_num])

        # reset random time
        random_time = [[delta for i in range(len(workload))] for i in range(edge_num)]
        # reset W
        W = [[100 for i in range(edge_num)] for i in range(edge_num)]

        # reset edge_computer_capblity
        edge_computer_cability = [max_cpu_capbility for i in range(edge_num)]

        # reset resouce upbound
        resouce_upbound = []

        for tmpava_bydevice in avatimelist:
            tmpsum = 0
            for tmpinterval in tmpava_bydevice:
                tmplen = tmpinterval[1] - tmpinterval[0]
                tmpsum = tmpsum + tmplen
            resouce_upbound.append(tmpsum)


        set_paramerters(workload=workload, datasize=datasize, pre=pre, succ=succ, num_edges=edge_num,window_size=window_size,
                        ava_time_list=avatimelist, random_time=random_time, bandwidth_edge=W,decision_time_list=decision_time_list,
                        taskindex2order_map=taskindex2order_map,order2taskindex_map=order2taskindex_map,order2subtaskindex_map=order2subtaskindex_map,
                        edge_computer_capability=edge_computer_cability, resouce_upbound=resouce_upbound)

        # tmptimelist = get_time_list()

        edge_num_time_list += get_run_time()

        # 将配置信息写入文件当中
        # with open("{0}_{1}_paramerters.txt".format(data_prefix, edge_num), 'a+') as file:
        #     file.write("\n")
        #     file.write("edge num: {0}\n avatime_list: {1}\n avatime_raido: {2}\n edge computer capability: {3}\n "
        #                "resource upbound: {4}\n n look: {5} random time: {6}\n bandwidth edge: {7} resouce upbound: {8}\n".format(
        #         edge_num, avatimelist, avatime_radio, edge_computer_cability, resouce_upbound, n_look, random_time, W, resouce_upbound
        #     ))

        # 实验数据记录
        for i in range(len(edge_num_time_list)):
            runtime_list[i].append([edge_num_time_list[i]])

    # 实验数据分算法记录
    runtime_dict = {}
    for i in range(len(runtime_list)):
        runtime_dict[ALGOR_NAME_LIST[i]] = runtime_list[i]

    return runtime_dict

# CPU处理速率
def exp_2_graph(taskgraphtype, expprefix):
    
    import pandas as pd
    from tqdm import tqdm
    time_dict_list = []
    # ratio_dict_list = []
    inter_num = defalut_inter_num
    new_max_cpu_capbility = 300

    # set big task graph paramerters
    task_info = get_connect_multiple_task_graph(defalut_request_number,taskgraphtype,tasktype=defalut_tasktype)
    
    for i in tqdm(range(inter_num)):
        tmptimedict = {}
        # tmpratiodict = {}

        for tmpalgorname in ALGOR_NAME_LIST:
            tmptimedict[tmpalgorname] = []
        
        # for tmpalgorname in ALGOR_NAME_LIST:
        #     tmpratiodict[tmpalgorname] = []

        decision_time_list,avatimelist = generate_randomtimeline(num_edges=defalut_edge_nums,
        start_ratio=defalut_start_ratio,start_sigma=defalut_start_sigma,
        ava_ratio=defalut_avaratio,ratio_sigma=defalut_sigma)



        for max_cpu_capbility in range(new_max_cpu_capbility, 600,30):
            # avatimelist = [generate_ava_time_by_jieduan(0.5, 20, 400, mu=5, sigma=5) for k in range(5)]
            # print("avatimelist:")
            # print(avatimelist)
            # print("decision_time_list:")
            # print(decision_time_list)
            # 不同算法的一次实验
            time_dict = taskgraph_exp("graph_iteration_{0}_{1}".format(i + 1, expprefix), taskgraphtype,
                                    task_info = task_info,
                                    max_edge_num=defalut_edge_nums, 
                                    avatimelist=avatimelist,
                                    decision_time_list=decision_time_list,
                                    max_cpu_capbility=max_cpu_capbility,
                                    change_edge_num=False)
            for tmpalgorname in ALGOR_NAME_LIST:
                tmptimedict[tmpalgorname].append(time_dict[tmpalgorname])
                # tmpratiodict[tmpalgorname].append(ratio_dict[tmpalgorname])

        time_dict_list.append(tmptimedict)
        # ratio_dict_list.append(tmpratiodict)

    avg_time_dict = result_timecal(time_dict_list,inter_num=inter_num)
    # avg_ratio_dict = result_ratiocal(ratio_dict_list,inter_num=inter_num)
    
    # print(avg_time_dict)
    df = pd.DataFrame(data=avg_time_dict)
    df.to_csv("{0}_{1}_{2}_{3}.csv".format('graph_iteration',str(inter_num),'max_time' ,expprefix), index=True)

    # df = pd.DataFrame(data=avg_ratio_dict)
    # df.to_csv("{0}_{1}_{2}_{3}.csv".format('graph_iteration',str(inter_num),'ratio' , expprefix), index=True)

# 传输随机扰动
def exp_3_graph(taskgraphtype, expprefix):
    import pandas as pd
    from tqdm import tqdm
    time_dict_list = []
    # ratio_dict_list = []

    inter_num = defalut_inter_num
    # avatime_radio = 0.5
    # simga = 0.1
    new_delta = defalut_delta 
    task_info = get_connect_multiple_task_graph(defalut_request_number,taskgraphtype,tasktype=defalut_tasktype)
    
    for i in tqdm(range(inter_num)):
        tmptimedict = {}
        # tmpratiodict = {}

        for tmpalgorname in ALGOR_NAME_LIST:
            tmptimedict[tmpalgorname] = []
        
        # for tmpalgorname in ALGOR_NAME_LIST:
        #     tmpratiodict[tmpalgorname] = []
        decision_time_list,avatimelist = generate_randomtimeline(num_edges=defalut_edge_nums,
        start_ratio=defalut_start_ratio,start_sigma=defalut_start_sigma,
        ava_ratio=defalut_avaratio,ratio_sigma=defalut_sigma)
        for t in range(0, 10):

            delta = new_delta + new_delta * t
            # avatimelist = [generate_ava_time_by_jieduan(0.5, 20, 400, mu=5, sigma=5) for k in range(5)]
            # print("avatimelist:")
            # print(avatimelist)
            # print("decision_time_list:")
            # print(decision_time_list)
            # 不同算法的一次实验
            time_dict = taskgraph_exp("graph_iteration_{0}_{1}".format(i + 1, expprefix), taskgraphtype,task_info=task_info,
                                    max_edge_num=defalut_edge_nums, 
                                    avatimelist=avatimelist,
                                    decision_time_list=decision_time_list,
                                    delta=delta,
                                    change_edge_num=False)
            for tmpalgorname in ALGOR_NAME_LIST:
                tmptimedict[tmpalgorname].append(time_dict[tmpalgorname])
                # tmpratiodict[tmpalgorname].append(ratio_dict[tmpalgorname])

        time_dict_list.append(tmptimedict)
        # ratio_dict_list.append(tmpratiodict)

    avg_time_dict = result_timecal(time_dict_list,inter_num=inter_num)
    # avg_ratio_dict = result_ratiocal(ratio_dict_list,inter_num=inter_num)
    
    # print(avg_time_dict)
    df = pd.DataFrame(data=avg_time_dict)
    df.to_csv("{0}_{1}_{2}_{3}.csv".format('graph_iteration',str(inter_num),'max_time' ,expprefix), index=True)

    # df = pd.DataFrame(data=avg_ratio_dict)
    # df.to_csv("{0}_{1}_{2}_{3}.csv".format('graph_iteration',str(inter_num),'ratio' , expprefix), index=True)

# 服务器数量影响
def exp_1_graph(taskgraphtype, expprefix):
    import pandas as pd
    from tqdm import tqdm
    time_dict_list = []
    # ratio_dict_list = []

    # avatime_radio = 0.5
    inter_num = defalut_inter_num
    task_info = get_connect_multiple_task_graph(defalut_request_number,taskgraphtype,tasktype=defalut_tasktype)
    # simga = 0.1
    # new_delta = 0.1

    for i in tqdm(range(inter_num)):
        tmptimedict = {}
        # tmpratiodict = {}

        for tmpalgorname in ALGOR_NAME_LIST:
            tmptimedict[tmpalgorname] = []
        
        # for tmpalgorname in ALGOR_NAME_LIST:
        #     tmpratiodict[tmpalgorname] = []

        
        # avatimelist = [generate_ava_time_by_jieduan(0.5, 20, 400, mu=5, sigma=5) for k in range(5)]
        # decision_time_list,avatimelist = generate_randomtimeline(defalut_edge_nums,
        # ava_ratio=defalut_avaratio,ava_sigma=defalut_sigma,
        # start_ratio=defalut_start_ratio,start_sigma=defalut_start_sigma)
        # print("avatimelist:")
        # print(avatimelist)
        # print("decision_time_list:")
        # print(decision_time_list)
        # 不同算法的一次实验

        time_dict = taskgraph_exp("graph_iteration_{0}_{1}".format(i + 1, expprefix), taskgraphtype,task_info=task_info,
                                change_edge_num=True)

        for tmpalgorname in ALGOR_NAME_LIST:
            tmptimedict[tmpalgorname] = [time_dict[tmpalgorname][i] for i in range(len(time_dict[tmpalgorname]))]
            # tmpratiodict[tmpalgorname] = [ratio_dict[tmpalgorname][i] for i in range(len(ratio_dict[tmpalgorname]))]

        time_dict_list.append(tmptimedict)
        # ratio_dict_list.append(tmpratiodict)

    avg_time_dict = result_timecal(time_dict_list,inter_num=inter_num)
    # avg_ratio_dict = result_ratiocal(ratio_dict_list,inter_num=inter_num)
    
    # print(avg_time_dict)
    df = pd.DataFrame(data=avg_time_dict)
    df.to_csv("{0}_{1}_{2}_{3}.csv".format('graph_iteration',str(inter_num),'max_time' ,expprefix), index=True)

    # df = pd.DataFrame(data=avg_ratio_dict)
    # df.to_csv("{0}_{1}_{2}_{3}.csv".format('graph_iteration',str(inter_num),'ratio' , expprefix), index=True)

# 服务器数量的运行时间
def exp_7_graph(taskgraphtype, expprefix):
    import pandas as pd
    from tqdm import tqdm
    runtime_dict_list = []
    
    inter_num = defalut_inter_num
    task_info = get_connect_multiple_task_graph(defalut_request_number,taskgraphtype,tasktype=defalut_tasktype)

    for i in tqdm(range(inter_num)):
        tmpruntimedict = {}

        for tmpalgorname in ALGOR_NAME_LIST:
            tmpruntimedict[tmpalgorname] = []

        # avatimelist = [generate_ava_time_by_jieduan(0.5, 20, 400, mu=5, sigma=5) for k in range(5)]
        # decision_time_list,avatimelist = generate_randomtimeline(defalut_edge_nums,ava_ratio=avatime_radio,sigma=simga)
        # print("avatimelist:")
        # print(avatimelist)
        # print("decision_time_list:")
        # print(decision_time_list)
        # 不同算法的一次实验

        runtime_dict = taskgraph_exp_runtime("graph_iteration_{0}_{1}".format(i + 1, expprefix), taskgraphtype,task_info=task_info,
                                change_edge_num=True)

        for tmpalgorname in ALGOR_NAME_LIST:
            tmpruntimedict[tmpalgorname] = [runtime_dict[tmpalgorname][i] for i in range(len(runtime_dict[tmpalgorname]))]

        runtime_dict_list.append(tmpruntimedict)

    avg_runtime_dict = result_timecal(runtime_dict_list,inter_num=inter_num)
    
    # print(avg_time_dict)
    df = pd.DataFrame(data=avg_runtime_dict)
    df.to_csv("{0}_{1}_{2}_{3}.csv".format('graph_iteration',str(inter_num),'run_time' ,expprefix), index=True)

# 任务比例(待)
def exp_41_graph(taskgraphtype, expprefix):
    import pandas as pd
    from tqdm import tqdm
    time_dict_list = []
    ratio_dict_list = []
    
    inter_num = defalut_inter_num
    # start_time_sigma = 0.001
    # start_time_ratio = 0.001
    request_number = [20,10,10]
    task_info = get_connect_multiple_task_graph(request_number,taskgraphtype,tasktype=defalut_tasktype)
    for i in tqdm(range(inter_num)):
        tmptimedict = {}
        # tmpratiodict = {}

        for tmpalgorname in ALGOR_NAME_LIST:
            tmptimedict[tmpalgorname] = []
        
        # for tmpalgorname in ALGOR_NAME_LIST:
        #     tmpratiodict[tmpalgorname] = []

        
        # avatimelist = [generate_ava_time_by_jieduan(0.5, 20, 400, mu=5, sigma=5) for k in range(5)]
        # decision_time_list,avatimelist = generate_randomtimeline(defalut_edge_nums,ava_ratio=avatime_radio,sigma=simga)
        # print("avatimelist:")
        # print(avatimelist)
        # print("decision_time_list:")
        # print(decision_time_list)
        # 不同算法的一次实验
        time_dict = taskgraph_exp("graph_iteration_{0}_{1}".format(i + 1, expprefix), 
        taskgraphtype,task_info=task_info,
                                change_edge_num=True)

        for tmpalgorname in ALGOR_NAME_LIST:
            tmptimedict[tmpalgorname] = [time_dict[tmpalgorname][i] for i in range(len(time_dict[tmpalgorname]))]
            # tmpratiodict[tmpalgorname] = [ratio_dict[tmpalgorname][i] for i in range(len(ratio_dict[tmpalgorname]))]

        time_dict_list.append(tmptimedict)
        # ratio_dict_list.append(tmpratiodict)

    avg_time_dict = result_timecal(time_dict_list,inter_num=inter_num)
    # avg_ratio_dict = result_ratiocal(ratio_dict_list,inter_num=inter_num)
    
    # print(avg_time_dict)
    df = pd.DataFrame(data=avg_time_dict)
    df.to_csv("{0}_{1}_{2}_{3}.csv".format('graph_iteration',str(inter_num),'max_time' ,expprefix), index=True)

    # df = pd.DataFrame(data=avg_ratio_dict)
    # df.to_csv("{0}_{1}_{2}_{3}.csv".format('graph_iteration',str(inter_num),'ratio' , expprefix), index=True)

def exp_42_graph(taskgraphtype, expprefix):
    import pandas as pd
    from tqdm import tqdm
    time_dict_list = []
    ratio_dict_list = []
    
    inter_num = defalut_inter_num
    
    request_number = [10,20,10]
    task_info = get_connect_multiple_task_graph(request_number,taskgraphtype,tasktype=defalut_tasktype)
    for i in tqdm(range(inter_num)):
        tmptimedict = {}
        # tmpratiodict = {}

        for tmpalgorname in ALGOR_NAME_LIST:
            tmptimedict[tmpalgorname] = []
        
        # for tmpalgorname in ALGOR_NAME_LIST:
        #     tmpratiodict[tmpalgorname] = []

        
        # avatimelist = [generate_ava_time_by_jieduan(0.5, 20, 400, mu=5, sigma=5) for k in range(5)]
        # decision_time_list,avatimelist = generate_randomtimeline(defalut_edge_nums,ava_ratio=avatime_radio,sigma=simga)
        # print("avatimelist:")
        # print(avatimelist)
        # print("decision_time_list:")
        # print(decision_time_list)
        # 不同算法的一次实验
        time_dict = taskgraph_exp("graph_iteration_{0}_{1}".format(i + 1, expprefix), 
        taskgraphtype,task_info=task_info,
                                change_edge_num=True)

        for tmpalgorname in ALGOR_NAME_LIST:
            tmptimedict[tmpalgorname] = [time_dict[tmpalgorname][i] for i in range(len(time_dict[tmpalgorname]))]
            # tmpratiodict[tmpalgorname] = [ratio_dict[tmpalgorname][i] for i in range(len(ratio_dict[tmpalgorname]))]

        time_dict_list.append(tmptimedict)
        # ratio_dict_list.append(tmpratiodict)

    avg_time_dict = result_timecal(time_dict_list,inter_num=inter_num)
    # avg_ratio_dict = result_ratiocal(ratio_dict_list,inter_num=inter_num)
    
    # print(avg_time_dict)
    df = pd.DataFrame(data=avg_time_dict)
    df.to_csv("{0}_{1}_{2}_{3}.csv".format('graph_iteration',str(inter_num),'max_time' ,expprefix), index=True)

    # df = pd.DataFrame(data=avg_ratio_dict)
    # df.to_csv("{0}_{1}_{2}_{3}.csv".format('graph_iteration',str(inter_num),'ratio' , expprefix), index=True)

def exp_43_graph(taskgraphtype, expprefix):
    import pandas as pd
    from tqdm import tqdm
    time_dict_list = []
    ratio_dict_list = []
    
    inter_num = defalut_inter_num
    # start_time_sigma = 0.001
    # start_time_ratio = 0.001
    request_number = [10,10,20]
    task_info = get_connect_multiple_task_graph(request_number,taskgraphtype,tasktype=defalut_tasktype)
    for i in tqdm(range(inter_num)):
        tmptimedict = {}
        # tmpratiodict = {}

        for tmpalgorname in ALGOR_NAME_LIST:
            tmptimedict[tmpalgorname] = []
        
        # for tmpalgorname in ALGOR_NAME_LIST:
        #     tmpratiodict[tmpalgorname] = []

        
        # avatimelist = [generate_ava_time_by_jieduan(0.5, 20, 400, mu=5, sigma=5) for k in range(5)]
        # decision_time_list,avatimelist = generate_randomtimeline(defalut_edge_nums,ava_ratio=avatime_radio,sigma=simga)
        # print("avatimelist:")
        # print(avatimelist)
        # print("decision_time_list:")
        # print(decision_time_list)
        # 不同算法的一次实验
        time_dict = taskgraph_exp("graph_iteration_{0}_{1}".format(i + 1, expprefix), 
        taskgraphtype,task_info=task_info,
                                change_edge_num=True)

        for tmpalgorname in ALGOR_NAME_LIST:
            tmptimedict[tmpalgorname] = [time_dict[tmpalgorname][i] for i in range(len(time_dict[tmpalgorname]))]
            # tmpratiodict[tmpalgorname] = [ratio_dict[tmpalgorname][i] for i in range(len(ratio_dict[tmpalgorname]))]

        time_dict_list.append(tmptimedict)
        # ratio_dict_list.append(tmpratiodict)

    avg_time_dict = result_timecal(time_dict_list,inter_num=inter_num)
    # avg_ratio_dict = result_ratiocal(ratio_dict_list,inter_num=inter_num)
    
    # print(avg_time_dict)
    df = pd.DataFrame(data=avg_time_dict)
    df.to_csv("{0}_{1}_{2}_{3}.csv".format('graph_iteration',str(inter_num),'max_time' ,expprefix), index=True)

    # df = pd.DataFrame(data=avg_ratio_dict)
    # df.to_csv("{0}_{1}_{2}_{3}.csv".format('graph_iteration',str(inter_num),'ratio' , expprefix), index=True)

# 有效时间比例
def exp_5_graph(taskgraphtype, expprefix):
    import pandas as pd
    from tqdm import tqdm
    time_dict_list = []
    ratio_dict_list = []

    inter_num = defalut_inter_num
    new_avatime_ratio = 70
    new_max_avatime_ratio = 75
    # simga = 0.1
    task_info = get_connect_multiple_task_graph(defalut_request_number,taskgraphtype,tasktype=defalut_tasktype)
    for i in tqdm(range(inter_num)):
        tmptimedict = {}
        # tmpratiodict = {}

        for tmpalgorname in ALGOR_NAME_LIST:
            tmptimedict[tmpalgorname] = []
        
        # for tmpalgorname in ALGOR_NAME_LIST:
        #     tmpratiodict[tmpalgorname] = []

        for avatime_ratio in range(new_avatime_ratio, new_max_avatime_ratio,1):
            avatime_ratio = avatime_ratio/100
            # avatimelist = [generate_ava_time_by_jieduan(0.5, 20, 400, mu=5, sigma=5) for k in range(5)]
            decision_time_list,avatimelist = generate_randomtimeline(num_edges=defalut_edge_nums,
        start_ratio=defalut_start_ratio,start_sigma=defalut_start_sigma,
        ava_ratio=avatime_ratio,ratio_sigma=defalut_sigma)
            # print("avatimelist:")
            # print(avatimelist)
            # print("decision_time_list:")
            # print(decision_time_list)
            # 不同算法的一次实验
            time_dict = taskgraph_exp("graph_iteration_{0}_{1}".format(i + 1, expprefix), taskgraphtype,task_info=task_info,
                                    max_edge_num=defalut_edge_nums, 
                                    avatimelist=avatimelist,
                                    decision_time_list=decision_time_list,
                                    change_edge_num=False)
            for tmpalgorname in ALGOR_NAME_LIST:
                tmptimedict[tmpalgorname].append(time_dict[tmpalgorname])
                # tmpratiodict[tmpalgorname].append(ratio_dict[tmpalgorname])

        time_dict_list.append(tmptimedict)
        # ratio_dict_list.append(tmpratiodict)

    avg_time_dict = result_timecal(time_dict_list,inter_num=inter_num)
    # avg_ratio_dict = result_ratiocal(ratio_dict_list,inter_num=inter_num)
    
    # print(avg_time_dict)
    df = pd.DataFrame(data=avg_time_dict)
    df.to_csv("{0}_{1}_{2}_{3}.csv".format('graph_iteration',str(inter_num),'max_time' ,expprefix), index=True)

    # df = pd.DataFrame(data=avg_ratio_dict)
    # df.to_csv("{0}_{1}_{2}_{3}.csv".format('graph_iteration',str(inter_num),'ratio' , expprefix), index=True)

# private task interval分布--稳定性
def exp_61_graph(taskgraphtype, expprefix):
    import pandas as pd
    from tqdm import tqdm
    time_dict_list = []
    ratio_dict_list = []
    
    inter_num = defalut_inter_num
    start_time_sigma = 0.1
    start_time_ratio = 0.01
    task_info = get_connect_multiple_task_graph(defalut_request_number,taskgraphtype,tasktype=defalut_tasktype)
    for i in tqdm(range(inter_num)):
        tmptimedict = {}
        # tmpratiodict = {}

        for tmpalgorname in ALGOR_NAME_LIST:
            tmptimedict[tmpalgorname] = []
        
        # for tmpalgorname in ALGOR_NAME_LIST:
        #     tmpratiodict[tmpalgorname] = []

        
        # avatimelist = [generate_ava_time_by_jieduan(0.5, 20, 400, mu=5, sigma=5) for k in range(5)]
        # decision_time_list,avatimelist = generate_randomtimeline(defalut_edge_nums,ava_ratio=avatime_radio,sigma=simga)
        # print("avatimelist:")
        # print(avatimelist)
        # print("decision_time_list:")
        # print(decision_time_list)
        # 不同算法的一次实验
        time_dict = taskgraph_exp("graph_iteration_{0}_{1}".format(i + 1, expprefix), 
        taskgraphtype,start_ratio=start_time_ratio,start_sigma=start_time_sigma,task_info=task_info,
                                change_edge_num=True)

        for tmpalgorname in ALGOR_NAME_LIST:
            tmptimedict[tmpalgorname] = [time_dict[tmpalgorname][i] for i in range(len(time_dict[tmpalgorname]))]
            # tmpratiodict[tmpalgorname] = [ratio_dict[tmpalgorname][i] for i in range(len(ratio_dict[tmpalgorname]))]

        time_dict_list.append(tmptimedict)
        # ratio_dict_list.append(tmpratiodict)

    avg_time_dict = result_timecal(time_dict_list,inter_num=inter_num)
    # avg_ratio_dict = result_ratiocal(ratio_dict_list,inter_num=inter_num)
    
    # print(avg_time_dict)
    df = pd.DataFrame(data=avg_time_dict)
    df.to_csv("{0}_{1}_{2}_{3}.csv".format('graph_iteration',str(inter_num),'max_time' ,expprefix), index=True)

    # df = pd.DataFrame(data=avg_ratio_dict)
    # df.to_csv("{0}_{1}_{2}_{3}.csv".format('graph_iteration',str(inter_num),'ratio' , expprefix), index=True)

def exp_62_graph(taskgraphtype, expprefix):
    import pandas as pd
    from tqdm import tqdm
    time_dict_list = []
    ratio_dict_list = []
    
    inter_num = defalut_inter_num
    start_time_sigma = 0.1
    start_time_ratio = 0.1
    task_info = get_connect_multiple_task_graph(defalut_request_number,taskgraphtype,tasktype=defalut_tasktype)
    for i in tqdm(range(inter_num)):
        tmptimedict = {}
        # tmpratiodict = {}

        for tmpalgorname in ALGOR_NAME_LIST:
            tmptimedict[tmpalgorname] = []
        
        # for tmpalgorname in ALGOR_NAME_LIST:
        #     tmpratiodict[tmpalgorname] = []

        
        # avatimelist = [generate_ava_time_by_jieduan(0.5, 20, 400, mu=5, sigma=5) for k in range(5)]
        # decision_time_list,avatimelist = generate_randomtimeline(defalut_edge_nums,ava_ratio=avatime_radio,sigma=simga)
        # print("avatimelist:")
        # print(avatimelist)
        # print("decision_time_list:")
        # print(decision_time_list)
        # 不同算法的一次实验
        time_dict = taskgraph_exp("graph_iteration_{0}_{1}".format(i + 1, expprefix), 
        taskgraphtype,start_ratio=start_time_ratio,start_sigma=start_time_sigma,task_info=task_info,
                                change_edge_num=True)

        for tmpalgorname in ALGOR_NAME_LIST:
            tmptimedict[tmpalgorname] = [time_dict[tmpalgorname][i] for i in range(len(time_dict[tmpalgorname]))]
            # tmpratiodict[tmpalgorname] = [ratio_dict[tmpalgorname][i] for i in range(len(ratio_dict[tmpalgorname]))]

        time_dict_list.append(tmptimedict)
        # ratio_dict_list.append(tmpratiodict)

    avg_time_dict = result_timecal(time_dict_list,inter_num=inter_num)
    # avg_ratio_dict = result_ratiocal(ratio_dict_list,inter_num=inter_num)
    
    # print(avg_time_dict)
    df = pd.DataFrame(data=avg_time_dict)
    df.to_csv("{0}_{1}_{2}_{3}.csv".format('graph_iteration',str(inter_num),'max_time' ,expprefix), index=True)

    # df = pd.DataFrame(data=avg_ratio_dict)
    # df.to_csv("{0}_{1}_{2}_{3}.csv".format('graph_iteration',str(inter_num),'ratio' , expprefix), index=True)

def exp_63_graph(taskgraphtype, expprefix):
    import pandas as pd
    from tqdm import tqdm
    time_dict_list = []
    ratio_dict_list = []
    
    inter_num = defalut_inter_num
    start_time_sigma = 0.2
    start_time_ratio = 0.1
    task_info = get_connect_multiple_task_graph(defalut_request_number,taskgraphtype,tasktype=defalut_tasktype)

    for i in tqdm(range(inter_num)):
        tmptimedict = {}
        # tmpratiodict = {}

        for tmpalgorname in ALGOR_NAME_LIST:
            tmptimedict[tmpalgorname] = []
        
        # for tmpalgorname in ALGOR_NAME_LIST:
        #     tmpratiodict[tmpalgorname] = []

        
        # avatimelist = [generate_ava_time_by_jieduan(0.5, 20, 400, mu=5, sigma=5) for k in range(5)]
        # decision_time_list,avatimelist = generate_randomtimeline(defalut_edge_nums,ava_ratio=avatime_radio,sigma=simga)
        # print("avatimelist:")
        # print(avatimelist)
        # print("decision_time_list:")
        # print(decision_time_list)
        # 不同算法的一次实验
        time_dict = taskgraph_exp("graph_iteration_{0}_{1}".format(i + 1, expprefix), 
        taskgraphtype,start_ratio=start_time_ratio,start_sigma=start_time_sigma,task_info=task_info,
                                change_edge_num=True)

        for tmpalgorname in ALGOR_NAME_LIST:
            tmptimedict[tmpalgorname] = [time_dict[tmpalgorname][i] for i in range(len(time_dict[tmpalgorname]))]
            # tmpratiodict[tmpalgorname] = [ratio_dict[tmpalgorname][i] for i in range(len(ratio_dict[tmpalgorname]))]

        time_dict_list.append(tmptimedict)
        # ratio_dict_list.append(tmpratiodict)

    avg_time_dict = result_timecal(time_dict_list,inter_num=inter_num)
    # avg_ratio_dict = result_ratiocal(ratio_dict_list,inter_num=inter_num)
    
    # print(avg_time_dict)
    df = pd.DataFrame(data=avg_time_dict)
    df.to_csv("{0}_{1}_{2}_{3}.csv".format('graph_iteration',str(inter_num),'max_time' ,expprefix), index=True)

    # df = pd.DataFrame(data=avg_ratio_dict)
    # df.to_csv("{0}_{1}_{2}_{3}.csv".format('graph_iteration',str(inter_num),'ratio' , expprefix), index=True)

# look_ahead window size 1-10
def exp_8_graph(taskgraphtype, expprefix):
    import pandas as pd
    from tqdm import tqdm
    time_dict_list = []
    ratio_dict_list = []

    inter_num = defalut_inter_num
    new_window_radio = 100
    new_max_window_radio = 500
    # simga = 0.1
    task_info = get_connect_multiple_task_graph(defalut_request_number,taskgraphtype,tasktype=defalut_tasktype)
    for i in tqdm(range(inter_num)):
        tmptimedict = {}
        # tmpratiodict = {}

        for tmpalgorname in ALGOR_NAME_LIST:
            tmptimedict[tmpalgorname] = []
        
        # for tmpalgorname in ALGOR_NAME_LIST:
        #     tmpratiodict[tmpalgorname] = []
        decision_time_list,avatimelist = generate_randomtimeline(num_edges=defalut_edge_nums,
                start_ratio=defalut_start_ratio,start_sigma=defalut_start_sigma,
                ava_ratio=defalut_avaratio,ratio_sigma=defalut_sigma)

        for window_radio in range(new_window_radio, new_max_window_radio,40):
            window_size = window_radio/1000 * default_timewindow
            # avatimelist = [generate_ava_time_by_jieduan(0.5, 20, 400, mu=5, sigma=5) for k in range(5)]
            
            # print("avatimelist:")
            # print(avatimelist)
            # print("decision_time_list:")
            # print(decision_time_list)
            # 不同算法的一次实验
            time_dict = taskgraph_exp("graph_iteration_{0}_{1}".format(i + 1, expprefix), taskgraphtype,task_info=task_info,
                                    max_edge_num=defalut_edge_nums, 
                                    avatimelist=avatimelist,
                                    decision_time_list=decision_time_list,
                                    window_size=window_size,
                                    change_edge_num=False)
            for tmpalgorname in ALGOR_NAME_LIST:
                tmptimedict[tmpalgorname].append(time_dict[tmpalgorname])
                # tmpratiodict[tmpalgorname].append(ratio_dict[tmpalgorname])

        time_dict_list.append(tmptimedict)
        # ratio_dict_list.append(tmpratiodict)

    avg_time_dict = result_timecal(time_dict_list,inter_num=inter_num)
    # avg_ratio_dict = result_ratiocal(ratio_dict_list,inter_num=inter_num)
    
    # print(avg_time_dict)
    df = pd.DataFrame(data=avg_time_dict)
    df.to_csv("{0}_{1}_{2}_{3}.csv".format('graph_iteration',str(inter_num),'max_time' ,expprefix), index=True)

    # df = pd.DataFrame(data=avg_ratio_dict)
    # df.to_csv("{0}_{1}_{2}_{3}.csv".format('graph_iteration',str(inter_num),'ratio' , expprefix), index=True)

# throughout with the increase of Te 25-30
def exp_9_graph(taskgraphtype, expprefix):
    import pandas as pd
    from tqdm import tqdm
    time_dict_list = []
    ratio_dict_list = []

    inter_num = defalut_inter_num
    new_timewindow = 10
    new_max_timewindow = 40
    # simga = 0.1
    task_info = get_connect_multiple_task_graph(defalut_request_number,taskgraphtype,tasktype=defalut_tasktype)
    for i in tqdm(range(inter_num)):
        tmptimedict = {}
        # tmpratiodict = {}

        for tmpalgorname in ALGOR_NAME_LIST:
            tmptimedict[tmpalgorname] = []
        
        # for tmpalgorname in ALGOR_NAME_LIST:
        #     tmpratiodict[tmpalgorname] = []
        for time_window in range(new_timewindow, new_max_timewindow,10):
        # for time_window in range(5):
            # time_window = time_window
            decision_time_list,avatimelist = generate_randomtimeline(num_edges=defalut_edge_nums,
                timewindow=time_window,timenumber=10,
                start_ratio=defalut_start_ratio,start_sigma=defalut_start_sigma,
                ava_ratio=defalut_avaratio,ratio_sigma=defalut_sigma)
            
            # avatimelist = [generate_ava_time_by_jieduan(0.5, 20, 400, mu=5, sigma=5) for k in range(5)]
            
            # print("avatimelist:")
            # print(avatimelist)
            # print("decision_time_list:")
            # print(decision_time_list)
            # 不同算法的一次实验
            time_dict = taskgraph_exp("graph_iteration_{0}_{1}".format(i + 1, expprefix), taskgraphtype,task_info=task_info,
                                    max_edge_num=defalut_edge_nums, 
                                    avatimelist=avatimelist,
                                    decision_time_list=decision_time_list,
                                    change_edge_num=False)
            for tmpalgorname in ALGOR_NAME_LIST:
                tmptimedict[tmpalgorname].append(time_dict[tmpalgorname])
                # tmpratiodict[tmpalgorname].append(ratio_dict[tmpalgorname])

        time_dict_list.append(tmptimedict)
        # ratio_dict_list.append(tmpratiodict)

    avg_time_dict = result_timecal(time_dict_list,inter_num=inter_num)
    # avg_ratio_dict = result_ratiocal(ratio_dict_list,inter_num=inter_num)
    
    # print(avg_time_dict)
    df = pd.DataFrame(data=avg_time_dict)
    df.to_csv("{0}_{1}_{2}_{3}.csv".format('graph_iteration',str(inter_num),'max_time' ,expprefix), index=True)

    # df = pd.DataFrame(data=avg_ratio_dict)
    # df.to_csv("{0}_{1}_{2}_{3}.csv".format('graph_iteration',str(inter_num),'ratio' , expprefix), index=True)


def exp2():
    # exp_2_graph("a", 'exp2_grapha')
    exp_2_graph(['ResNet18','Vgg16','AlexNet'],'exp2_graph_Connect')
    # exp_2_graph('ResNet18','exp2_graph_ResNet18')
    # exp_2_graph('Vgg16','exp2_graph_Vgg16')
    # exp_2_graph('Inceptionv3','exp2_graph_Inceptionv3')
    # exp_2_graph(2, 'exp2_graphb')
    # exp_2_graph(3, 'exp2_graphc')
    # exp_2_graph(4, 'exp2_graphd')
    # exp_2_graph(5, 'exp2_graphe')

def exp1():
    # exp_1_graph("a", 'exp1_grapha')
    exp_1_graph(['ResNet18','Vgg16','AlexNet'],'exp1_graph_Connect')
    # exp_1_graph('ResNet18','exp1_graph_ResNet18')
    # exp_1_graph('Vgg16','exp1_graph_Vgg16')
    # exp_1_graph('Inceptionv3','exp1_graph_Inceptionv3')

    # exp_1_graph(2, 'exp1_graphb')
    # exp_1_graph(3, 'exp1_graphc')
    # exp_1_graph(4, 'exp1_graphd')
    # exp_1_graph(5, 'exp1_graphe')

def exp3():
    # exp_3_graph("a", 'exp3_grapha')
    exp_3_graph(['ResNet18','Vgg16','AlexNet'],'exp3_graph_Connect')
    # exp_3_graph('ResNet18','exp3_graph_ResNet18')
    # exp_3_graph('Vgg16','exp3_graph_Vgg16')
    # exp_3_graph('Inceptionv3','exp3_graph_Inceptionv3')
    # exp_3_graph(2, 'exp3_graphb')
    # exp_3_graph(3, 'exp3_graphc')
    # exp_3_graph(4, 'exp3_graphd')
    # exp_3_graph(5, 'exp3_graphe')

def exp4():
    # exp_4_graph("a", 'exp4_grapha')
    exp_41_graph(['ResNet18','Vgg16','AlexNet'],'exp41_graph_Connect')
    exp_42_graph(['ResNet18','Vgg16','AlexNet'],'exp42_graph_Connect')
    exp_43_graph(['ResNet18','Vgg16','AlexNet'],'exp43_graph_Connect')
    # exp_4_graph('ResNet18','exp4_graph_ResNet18')
    # exp_4_graph('Vgg16','exp4_graph_Vgg16')
    # exp_4_graph('Inceptionv3','exp4_graph_Inceptionv3')
    # exp_4_graph(2, 'exp4_graphb')
    # exp_4_graph(3, 'exp4_graphc')
    # exp_4_graph(4, 'exp4_graphd')
    # exp_4_graph(5, 'exp4_graphe')

def exp5():
    # exp_5_graph("a", 'exp5_grapha')
    exp_5_graph(['ResNet18','Vgg16','AlexNet'],'exp5_graph_Connect')
    # exp_5_graph('ResNet18','exp5_graph_ResNet18')
    # exp_5_graph('Vgg16','exp5_graph_Vgg16')
    # exp_5_graph('Inceptionv3','exp5_graph_Inceptionv3')
    # exp_5_graph(2, 'exp5_graphb')
    # exp_5_graph(3, 'exp5_graphc')
    # exp_5_graph(4, 'exp5_graphd')
    # exp_5_graph(5, 'exp5_graphe')

def exp6():
    # exp_61_graph("a", 'exp61_grapha')
    exp_61_graph(['ResNet18','Vgg16','AlexNet'],'exp61_graph_Connect')
    # exp_61_graph('ResNet18','exp61_graph_ResNet18')
    # exp_61_graph('Vgg16','exp61_graph_Vgg16')
    # exp_61_graph('Inceptionv3','exp61_graph_Inceptionv3')
    # exp_61_graph(2, 'exp61_graphb')
    # exp_61_graph(3, 'exp61_graphc')
    # exp_61_graph(4, 'exp61_graphd')
    # exp_61_graph(5, 'exp61_graphe')

    # exp_62_graph("a", 'exp62_grapha')
    exp_62_graph(['ResNet18','Vgg16','AlexNet'],'exp62_graph_Connect')
    # exp_62_graph('ResNet18','exp62_graph_ResNet18')
    # exp_62_graph('Vgg16','exp62_graph_Vgg16')
    # exp_62_graph('Inceptionv3','exp62_graph_Inceptionv3')
    # exp_62_graph(2, 'exp62_graphb')
    # exp_62_graph(3, 'exp62_graphc')
    # exp_62_graph(4, 'exp62_graphd')
    # exp_62_graph(5, 'exp62_graphe')

    # exp_63_graph("a", 'exp63_grapha')
    exp_63_graph(['ResNet18','Vgg16','AlexNet'],'exp63_graph_Connect')
    # exp_63_graph('ResNet18','exp63_graph_ResNet18')
    # exp_63_graph('Vgg16','exp63_graph_Vgg16')
    # exp_63_graph('Inceptionv3','exp63_graph_Inceptionv3')
    # exp_63_graph(2, 'exp63_graphb')
    # exp_63_graph(3, 'exp63_graphc')
    # exp_63_graph(4, 'exp63_graphd')
    # exp_63_graph(5, 'exp63_graphe')

def exp7():
    # exp_7_graph("a", 'exp7_grapha')
    exp_7_graph(['ResNet18','Vgg16','AlexNet'],'exp7_graph_Connect')
    # exp_7_graph('ResNet18','exp7_graph_ResNet18')
    # exp_7_graph('Vgg16','exp7_graph_Vgg16')
    # exp_7_graph('Inceptionv3','exp7_graph_Inceptionv3')
    # exp_7_graph(2, 'exp7_graphb')
    # exp_7_graph(3, 'exp7_graphc')
    # exp_7_graph(4, 'exp7_graphd')
    # exp_7_graph(5, 'exp7_graphe')

def exp8():
    exp_8_graph(['ResNet18','Vgg16','AlexNet'],'exp8_graph_Connect')


def exp9():
    exp_9_graph(['ResNet18','Vgg16','AlexNet'],'exp9_graph_Connect')


if __name__ == "__main__":
    # exp1()
    # exp2()
    # exp3()
    # exp4()
    # exp5()
    # exp6()
    # exp7()
    exp8()
    # exp9()