from Taskgraph_pre import GRAPH_PRE_A,GRAPH_PRE_ResNet18,GRAPH_PRE_Vgg16,GRAPH_PRE_Inceptionv3,GRAPH_PRE_AlexNet

decistion_time_number = 10
default_timewindow = 30


lookahead_window_size = default_timewindow

# --------------任务合成---------------
# task_dict_list = []
# 1:max workload 2:min workload 3: fcfs 4 max budget 5 min budget
# order_mode = 3
# 根据不同规则(1:max workload 2: min workload 3: fcfs 4 budget) 对子任务编号和遍历
# taskindex2order_map,order2taskindex_map,order2subtaskindex_map = None,None,None
# visit_order = None

# 按一定顺序 先来先到的顺序 构造map
def connect_graph_map_from_pre(task_dict_list,visit_order):
    addnum = 1
    taskindex2order_map = [[] for i in range(len(task_dict_list))]
    sum_subtask_num = sum([len(task_dict_list[i]["workload"]) for i in range(len(task_dict_list))])
    order2taskindex_map = [-1 for i in range(sum_subtask_num + 2)]
    order2subtaskindex_map = [-1 for i in range(sum_subtask_num + 2)]
    subtask2task_map = [-1 for i in range(sum_subtask_num + 2)]
    
    for task_index in visit_order:
        taskindex2order_map[task_index] += [i + addnum for i in range(len(task_dict_list[task_index]["workload"]))]
        addnum += len(task_dict_list[task_index]["workload"])
    
    for task_index in visit_order:
        for i in range(len(taskindex2order_map[task_index])):
            subtask_index = taskindex2order_map[task_index][i]
            order2taskindex_map[subtask_index] = task_index

    for task_index in visit_order:
        for i in range(len(taskindex2order_map[task_index])):
            subtask_index = taskindex2order_map[task_index][i]
            order2subtaskindex_map[subtask_index] = i
 

    return taskindex2order_map,order2taskindex_map,order2subtaskindex_map

def connect_graph_workload_datasize(task_dict_list,visit_order):
    workload = []
    datasize = []

    workload.append(0)
    tmpdatasize = []
    for task_index in visit_order:
        tmpdatasize.append(task_dict_list[task_index]["input_datasize"])
    datasize.append(tmpdatasize) 
    
    for task_index in visit_order:
        workload += task_dict_list[task_index]["workload"]    
        datasize += task_dict_list[task_index]["datasize"]
    
    workload.append(0)
    datasize.append([0])
    
    return workload,datasize

# 按一定顺序 先来先到的顺序 编号和连接成大图
def connect_graph_succ(task_dict_list,visit_order):
    succ = []
    exit_node = []
    entry_node = []
    addnum = 1
    task_entry_node_list = []
    subtask_num = sum([len(task["workload"]) for task in task_dict_list]) + 1
    for task_index in visit_order:
        for i,task_succ in enumerate(task_dict_list[task_index]["succ"]):
            if len(task_succ) == 0:
                task_succ = [subtask_num]
                succ.append(task_succ)
            else:
                task_succ = [task_succ[i] + addnum for i in range(len(task_succ))]
                succ.append(task_succ)
        task_entry_node_list.append(addnum)
        addnum = len(succ) + 1

    succ.insert(0,task_entry_node_list)
    succ.append(exit_node)

    return succ

def connect_graph_pre(task_dict_list,visit_order):
    pre = []
    entry_node = []
    pre.append(entry_node)
    addnum = 1
    task_exit_node_list = []
    for task_index in visit_order:
        for task_pre in task_dict_list[task_index]["pre"]:
            if len(task_pre) == 0:
                task_pre = [0]
                pre.append(task_pre)
            else:
                task_pre = [task_pre[i] + addnum for i in range(len(task_pre))]
                pre.append(task_pre)
        task_exit_node_list.append(len(task_dict_list[task_index]["workload"]) - 1 + addnum)
        addnum = len(pre)
    exit_node = task_exit_node_list
    pre.append(exit_node)
    return pre

def select_order_from_mode(task_dict_list,order_mode=3):
    
    import numpy as np
    # 1:max workload 2:min workload 3: fcfs 4 max budget 5 min budget
    if order_mode == 1:
        # 获得任务负载按大到小的任务号序列
        task_workload_list = []
        for task_index,task in enumerate(task_dict_list):
            sum_task_worklaod = sum(task["workload"])
            task_workload_list.append(sum_task_worklaod)
        task_workload_list = np.array(task_workload_list)
        visit_order = np.argsort(-task_workload_list)

    elif order_mode == 2:
        # 获得任务负载按小到大的任务号序列
        task_workload_list = []
        for task_index,task in enumerate(task_dict_list):
            sum_task_worklaod = sum(task["workload"])
            task_workload_list.append(sum_task_worklaod)
        task_workload_list = np.array(task_workload_list)
        visit_order = np.argsort(task_workload_list)

    elif order_mode == 3:
        # 获得任务先来后到序列
        visit_order = [i for i in range(len(task_dict_list))]

    elif order_mode == 4:
        # 获得任务预算按大到小的任务号序列
        task_budget_list = []
        for task_index,task in enumerate(task_dict_list):
            task_budget_list.append(task["budget"])
        task_budget_list = np.array(task_budget_list)
        visit_order = np.argsort(-task_budget_list)

    elif order_mode == 5:
        # 获得任务负载按小到大的任务号序列
        task_budget_list = []
        for task_index,task in enumerate(task_dict_list):
            task_budget_list.append(task["budget"])
        task_budget_list = np.array(task_budget_list)
        visit_order = np.argsort(task_budget_list)
    return visit_order

def connect_graph(task_dict_list):

    visit_order = select_order_from_mode(task_dict_list)
    pre = connect_graph_pre(task_dict_list,visit_order)
    succ = connect_graph_succ(task_dict_list,visit_order)
    workload,datasize = connect_graph_workload_datasize(task_dict_list,visit_order)
    taskindex2order_map,order2taskindex_map,order2subtaskindex_map = connect_graph_map_from_pre(task_dict_list,visit_order)

    return pre,succ,workload,datasize,taskindex2order_map,order2taskindex_map,order2subtaskindex_map
# --------------任务合成---------------

# --------------任务外部读写合---------------
# 从文件得到任务的负载权重
def get_workload_and_datasize(filename,tasktype):
    import pandas as pd

    df = pd.read_csv(filename, delimiter=',')

    if tasktype == 0:
        # 对DNN数据进行规约化
        workload = df['GFLOPS'].tolist()
        datasize = df['assigned_memory_usage'].tolist()
        # 浮点次数 GFLOPS 
        datasize = [tmp for tmp in datasize]
        # 向量数据量 以一个float型存储 16位 2B 单位为B
        workload = [tmp * 2 for tmp in workload]
    else:
        # 对Google cluster数据进行规约化
        workload = df['CPU_rate'].tolist()
        datasize = df['assigned_memory_usage'].tolist()
        datasize = [tmp * 256 * 10 for tmp in datasize]
        workload = [tmp * 600 for tmp in workload]
    
    return workload, datasize

# 从前序得后序
def get_succ_from_pre(pre):
    succ = []

    for tmptask in range(len(pre)):
        task_succ = []

        for i,tmp in enumerate(pre):
            if tmptask in tmp:
                task_succ.append(i)
        succ.append(task_succ)

    return succ

# 设置任务图的权重负载参数
def get_task_graph_paramerters(taskgraphname,tasktype):
    '''
    根据任务图类型 获取任务图相关的参数
    * workload
    * datasize
    * pre
    * succ

    :param taskgraphname: 'a' 'b' 'c' 'd' 'e'
    :return: workload, datasize, pre, succ
    '''

    # get pre
    # graphname = ['a', 'b', 'c', 'd', 'e']
    if taskgraphname == 'a':
        pre = GRAPH_PRE_A.copy()
    if taskgraphname == 'b':
        pre = GRAPH_PRE_B
    if taskgraphname == 'c':
        pre = GRAPH_PRE_C
    if taskgraphname == 'd':
        pre = GRAPH_PRE_D
    if taskgraphname == 'e':
        pre = GRAPH_PRE_E

    if taskgraphname == "ResNet18":
        pre = GRAPH_PRE_ResNet18
    
    if taskgraphname == "Vgg16":
        pre = GRAPH_PRE_Vgg16

    if taskgraphname == "Inceptionv3":
        pre = GRAPH_PRE_Inceptionv3

    if taskgraphname == "AlexNet":
        pre = GRAPH_PRE_AlexNet

    tmpworkload, tmpdatasize = get_workload_and_datasize('graph_{0}.csv'.format(taskgraphname),tasktype)
    succ = get_succ_from_pre(pre)

    # reset the workload and datasize based on the task graph we set

    workload = tmpworkload
    datasize = [[0 for i in range(len(workload))] for j in range(len(workload))]

    for i in range(len(workload)):
        for j in range(len(workload)):
            if j in succ[i]:
                datasize[i][j] = tmpdatasize[i]

    return workload, datasize, pre, succ

# warning : input_datasize = 0
def get_connect_task_graph(request_number,taskgraph,tasktype):
    task_dict_list = []
    for i in range(request_number):
        task_dict = {}
        workload,datasize,pre,succ = get_task_graph_paramerters(taskgraph,tasktype)
        task_dict["pre"] = pre
        task_dict['succ'] = succ
        task_dict['workload'] = workload
        task_dict['datasize'] = datasize
        task_dict['input_datasize'] = 0
        task_dict_list.append(task_dict)
    return connect_graph(task_dict_list)

def get_connect_multiple_task_graph(request_number_list,taskgraph_list,tasktype):
    task_dict_list = []
    for i,taskgraph in enumerate(taskgraph_list):
        request_number =  request_number_list[i]
        tmp_tasktype = tasktype[i]
        for j in range(request_number):
            task_dict = {}
            workload,datasize,pre,succ = get_task_graph_paramerters(taskgraph,tmp_tasktype)
            task_dict["pre"] = pre
            task_dict['succ'] = succ
            task_dict['workload'] = workload
            task_dict['datasize'] = datasize
            task_dict['input_datasize'] = 0
            task_dict_list.append(task_dict)
    return connect_graph(task_dict_list)
    
# --------------任务外部读写合---------------

# --------------时间线生成器---------------
# 生成随机分布的得数
def get_normal_ratio(ava_ratio,sigma):
    from scipy.stats import truncnorm
    from scipy.stats import norm
    if ava_ratio == 0 and sigma == 0:
        return 0
    mu = ava_ratio
    # lower = mu - 2 * sigma
    upper = mu + 2 * sigma
    X = truncnorm(0, (upper - mu) / sigma, loc=mu, scale=sigma)
    # generate 1000 sample data
    samples = X.rvs(1)
    return samples[0]

# 在每台server生成随机有效时间线
def generate_randomtimeline(num_edges,start_ratio,start_sigma,timewindow=default_timewindow,timenumber=decistion_time_number,ratio_sigma=0.1,ava_ratio=0.5): 
    import random
    totaltime = timenumber * timewindow
    start_time = 0
    decision_time_list = []
    ava_time_list = []
    for j in range(num_edges):
        tmp_ava_time_list = []
        processor_timeline = []
        start_time = 0
        for i in range(round(totaltime/timewindow)):
            
            tmp_interval = []
            distribute_ava_ratio = get_normal_ratio(ava_ratio,ratio_sigma)
            distribute_start_ratio = get_normal_ratio(start_ratio,start_sigma)

            unavailable_time = (1-distribute_ava_ratio) * timewindow
            unavailable_start_time_related = (distribute_start_ratio) * timewindow

            # 在分布式学习中 前空闲出来的传输空闲 还有 后空闲出来的等待更新空闲

            front_available_starttime = start_time
            front_available_endtime = start_time + unavailable_start_time_related

            unavailable_start_time = front_available_endtime
            unavailable_end_time = 0

            if unavailable_start_time + unavailable_time > (i+1) * timewindow:
                unavailable_end_time = (i+1) * timewindow
            else:
                unavailable_end_time = unavailable_start_time + unavailable_time

            back_available_starttime = unavailable_end_time
            back_available_endtime = (i+1) * timewindow

            unavailable_time_tmp = [unavailable_start_time,unavailable_end_time]

            if front_available_starttime != front_available_endtime:
                front_available_time_tmp = [front_available_starttime,front_available_endtime]
                tmp_ava_time_list.append(front_available_time_tmp)
        
            if back_available_starttime != back_available_endtime:
                back_available_time_tmp = [back_available_starttime,back_available_endtime]
                tmp_ava_time_list.append(back_available_time_tmp)

            start_time = back_available_endtime

            processor_timeline.append(unavailable_time_tmp)

        decision_time_list.append(processor_timeline)

        ava_time_list.append(tmp_ava_time_list)
    return decision_time_list,ava_time_list

# --------------时间线生成器---------------
