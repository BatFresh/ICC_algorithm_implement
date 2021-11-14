'可视化调度结果'
def plot_result(answer_list, ava_time_list, **kwargs):
    import matplotlib.pyplot as plt
    import numpy as np
    import math

    # huge hight
    huge_hight = 3

    # huge margin
    huge_margin = 1

    alldata = []
    for i in range(len(ava_time_list)):
        for tmp in ava_time_list[i]:
            alldata += tmp
    max_time = np.max(alldata)
    num_process = len(ava_time_list)

    fig, ax = plt.subplots()

    # generate y lim
    y_lim = huge_hight * num_process * 2 + (num_process * 2 + 1) * huge_margin
    ax.set_ylim(0, y_lim)

    # generate y ticks
    y_ticks = [(i-1)*huge_hight+huge_hight/2+i*huge_margin for i in range(1, num_process*2+1)]
    ax.set_yticks(y_ticks)

    # set y lables
    y_labels = ['processor {0} {1}'.format(math.ceil(i/2), 'aviable' if i % 2 == 1 else 'schedule')  for i in range(1, num_process*2+1)]
    ax.set_yticklabels(y_labels)

    # set x lim
    ax.set_xlim(0, max_time)

    # generate x ticks
    x_ticks = []
    for i in range(len(answer_list)):
        for tmp in answer_list[i]:
            if tmp[1] not in x_ticks:
                x_ticks.append(tmp[1])
            if tmp[2] not in x_ticks:
                x_ticks.append(tmp[2])

        # for tmp in ava_time_list[i]:
        #     if tmp[0] not in x_ticks:
        #         x_ticks.append(tmp[0])
        #     if tmp[1] not in x_ticks:
        #         x_ticks.append(tmp[1])

    ax.set_xticks(x_ticks, minor=False)

    # set x label
    ax.set_xlabel("Time")

    ax.grid(True)
    text_color = 'white'

    if 'title' in kwargs:
        if type(kwargs['title']) != str:
            ax.set_title(str(kwargs['title']))
        else:
            ax.set_title(kwargs['title'])

    # for each processor: gnerate input data, figure, text
    for i in range(num_process):
        ava_data = []

        for tmp in ava_time_list[i]:
            ava_data.append((tmp[0], tmp[1]-tmp[0]))
        ax.broken_barh(ava_data, (y_ticks[i*2]-huge_hight/2, huge_hight), facecolors='tab:blue')

        answer_data = []
        for tmp in answer_list[i]:
            answer_data.append((tmp[1], tmp[2]-tmp[1]))

        ax.broken_barh(answer_data, (y_ticks[i*2+1]-huge_hight/2, huge_hight), facecolors='tab:red')

        for tmp in answer_list[i]:
            ax.text((tmp[1]+tmp[2])/2, y_ticks[i*2+1], '{0}'.format(tmp[0]), ha='center', va='center', color=text_color,
                    fontsize='x-large')

    plt.show()

from Dataset import get_connect_task_graph,generate_randomtimeline,get_connect_multiple_task_graph,lookahead_window_size

# task_dict_1 = {
#     "pre":[[], [0], [1], [2]],
#     "succ":[[1], [2], [3], []],
#     "deadline":50,
#     "workload":[6]*4,
#     "datasize":[[100 for i in range(4)] for j in range(4)],
#     "budget":700, #579.5999999999999 615
#     "input_datasize":0
# }
# task_dict_2 = {
#     "pre":[[], [0], [1], [2], [3]],
#     "succ":[[1], [2], [3], [4], []],
#     "deadline":50,
#     "workload":[10]*5,
#     "datasize":[[100 for i in range(5)] for j in range(5)],
#     "budget":1500,#453.7099999999999 1300
#     "input_datasize":0
# }
# task_dict_3 = {
#     "pre":[[], [0], [0], [1,2]],
#     "succ":[[1, 2], [3], [3], []],
#     "deadline":50,
#     "workload":[6]*4,
#     "datasize":[[100 for i in range(4)] for j in range(4)],
#     "budget":700,#554.9499999999998 615
#     "input_datasize":0
# }
# task_dict_4 = {
#     "pre":[[], [0], [0], [1,2]],
#     "succ":[[1, 2], [3], [3], []],
#     "deadline":50,
#     "workload":[6]*4,
#     "datasize":[[100 for i in range(4)] for j in range(4)],
#     "budget":700,#554.9499999999998 615
#     "input_datasize":0
# }
# task_dict_5 = {
#     "pre":[[], [0], [0], [1,2]],
#     "succ":[[1, 2], [3], [3], []],
#     "deadline":50,
#     "workload":[6]*4,
#     "datasize":[[100 for i in range(4)] for j in range(4)],
#     "budget":700,#554.9499999999998 615
#     "input_datasize":0
# }
# task_dict_6 = {
#     "pre":[[], [0], [1], [2], [3]],
#     "succ":[[1], [2], [3], [4], []],
#     "deadline":50,
#     "workload":[10]*5,
#     "datasize":[[100 for i in range(5)] for j in range(5)],
#     "budget":1500,#453.7099999999999 1300
#     "input_datasize":0
# }
# task_dict_7 = {
#     "pre":[[], [0], [1], [2], [3]],
#     "succ":[[1], [2], [3], [4], []],
#     "deadline":50,
#     "workload":[10]*5,
#     "datasize":[[100 for i in range(5)] for j in range(5)],
#     "budget":1500,#453.7099999999999 1300
#     "input_datasize":0
# }

# task_dict_list = [task_dict_4,task_dict_4,task_dict_4,task_dict_4,task_dict_4,task_dict_4,task_dict_4,task_dict_4]
# # 1:max workload 2:min workload 3: fcfs 4 max budget 5 min budget
# order_mode = 3
# # 根据不同规则(1:max workload 2: min workload 3: fcfs 4 budget) 对子任务编号和遍历
# taskindex2order_map,order2taskindex_map,order2subtaskindex_map = None,None,None
# visit_order = None
# pre = []
# succ = []
# workload = []
# datasize = []

# request_number = 20
defalut_tasktype = [0,0,0]
defalut_request_number = [10,10,10]

# pre,succ,workload,datasize,taskindex2order_map,order2taskindex_map,order2subtaskindex_map = get_connect_task_graph(request_number=request_number,taskgraph="Vgg16",tasktype=0)
pre,succ,workload,datasize,taskindex2order_map,order2taskindex_map,order2subtaskindex_map = get_connect_multiple_task_graph(defalut_request_number,['ResNet18','Vgg16','AlexNet'],tasktype=defalut_tasktype)
# device related parameters
edge_num = 10
W = [[12.5 for i in range(edge_num)] for j in range(edge_num)]
edge_computer_capability = [300 for i in range(edge_num)]
random_time = [[0 for i in range(len(workload))] for j in range(edge_num)]

# slot related parameters
from Dataset import default_timewindow,decistion_time_number

# decistion_time_number = 5
# default_timewindow = 0.02

default_totaltime = decistion_time_number * default_timewindow

window_size = default_timewindow

decision_time_list,ava_time_list = generate_randomtimeline(num_edges=edge_num,start_ratio=0.1,start_sigma=0.01,ratio_sigma=0.01,ava_ratio=0.5)

resouce_upbound = []
for tmpava_bydevice in ava_time_list:
    tmpsum = 0
    for tmpinterval in tmpava_bydevice:
        tmplen = tmpinterval[1] - tmpinterval[0]
        tmpsum = tmpsum + tmplen
    resouce_upbound.append(tmpsum)

def first_order_sequence(pre,succ):
    from queue import Queue

    visited_queue = Queue()
    visited = set()
    visited_sequence = [] #已被执行完的

    # find the input task
    first_task = -1
    for i in range(len(pre)):
        if len(pre[i]) == 0:
            first_task = i
            break
    if first_task == -1:
        print("Can not find the input task {0}".format(pre))
        exit(-1)

    visited_queue.put(first_task)

    while not visited_queue.empty():
        tmptask = visited_queue.get()

        # 判断是否所有前驱任务都在已被执行完的列表里面 如果在里面 才进行访问 否则 continue
        flag = True
        for tmppretask in pre[tmptask]:
            if tmppretask not in visited_sequence:
                flag = False
                break

        if flag:
            visited.add(tmptask)

            visited_sequence.append(tmptask)

            for tmp in succ[tmptask]:
                if tmp not in visited:
                    visited_queue.put(tmp)
                    visited.add(tmp)

        else:
            visited_queue.put(tmptask)

    return visited_sequence

def set_paramerters(**kwargs):

    if 'window_size' in kwargs:
        global window_size
        window_size = kwargs['window_size']

    if 'num_edges' in kwargs:
        global edge_num
        edge_num = kwargs['num_edges']

    if 'ava_time_list' in kwargs:
        global  ava_time_list
        ava_time_list = kwargs['ava_time_list']

    if 'decision_time_list' in kwargs:
        global  decision_time_list
        decision_time_list = kwargs['decision_time_list']

    if 'pre' in kwargs:
        global  pre
        pre = kwargs['pre']

    if 'succ' in kwargs:
        global succ
        succ = kwargs['succ']

    if 'datasize' in kwargs:
        global  datasize
        datasize = kwargs['datasize']

    if 'workload' in kwargs:
        global workload
        workload = kwargs['workload']

    if 'bandwidth_edge' in kwargs:
        global  W
        W = kwargs['bandwidth_edge']

    if 'edge_computer_capability' in kwargs:
        global  edge_computer_capability
        edge_computer_capability = kwargs['edge_computer_capability']

    if 'resouce_upbound' in kwargs:
        global  resouce_upbound
        resouce_upbound = kwargs['resouce_upbound']

    if 'random_time' in kwargs:
        global random_time
        random_time = kwargs['random_time']
    
    if 'taskindex2order_map' in kwargs:
        global taskindex2order_map
        taskindex2order_map = kwargs['taskindex2order_map']
    
    if 'order2taskindex_map' in kwargs:
        global order2taskindex_map
        order2taskindex_map = kwargs['order2taskindex_map']

    if 'order2subtaskindex_map' in kwargs:
        global order2subtaskindex_map
        order2subtaskindex_map = kwargs['order2subtaskindex_map']

def takeSecond(ele):
    return ele[1]

def takeFirst(ele):
    return ele[0]

def computer_eta(workload,edge_num):
    eta = max([max(workload), sum(workload)/edge_num])
    return eta

def computer_Cij(answers_list, task_i, task_j, task_j_excuted_processor):
    tmpdatasize = datasize[task_i][order2subtaskindex_map[task_j]]
    # find task i executed place
    task_i_executed_place = -1
    # subtask_index = taskindex2order_map[task_index][task_i] 
    for i in range(len(answers_list)):
        for j in range(len(answers_list[i])):
            if answers_list[i][j][0] == task_i:
                task_i_executed_place = i
                break

    if task_i_executed_place == -1:
        print("Cannot find presequence task {0} in answer".format(task_i))
        return None

    if task_j_excuted_processor == task_i_executed_place:
        return  0
    tmpw = W[task_i_executed_place][task_j_excuted_processor]
    return random_time[task_i_executed_place][task_j_excuted_processor] + tmpdatasize/tmpw

def get_algorithm_timelist():

    # # offline
    # greedy_time_reservation_ans = greedy_time_reservation()
    # # heft_time_reservation_ans = heft_time_reservation()
    # heft_ans = heft()
    # greedy_ans = greedy()
    # nsga_ans = NSGA()
    # online
    # print(window_size)
    greedy_time_reservation_nlook_back_ans = greedy_time_reservation_nlook_back(window_size)
    heft_n_look_back_ans = heft_n_look_back(window_size)
    greedy_nlook_back_ans = greedy_nlook_back(window_size)
    NSGA_n_look_back_ans = NSGA_n_look_back(window_size)
    # # online improved
    # greedy_nlook_back_improved_ans = greedy_nlook_back_improved(window_size)
    # greedy_time_reservation_nlook_back_improved_ans = greedy_time_reservation_nlook_back_improved(window_size)
    # heft_n_look_back_improved_ans = heft_n_look_back_improved(window_size)

    # 
    # 
    # 
    # benchmark 
    # greedy_time_reservation_ans,heft_ans,greedy_ans,nsga_ans
    # greedy_time_reservation_nlook_back_improved_ans,heft_n_look_back_improved_ans,greedy_nlook_back_improved_ans,
    return [
    greedy_time_reservation_nlook_back_ans,heft_n_look_back_ans,greedy_nlook_back_ans,NSGA_n_look_back_ans
    
    ]
    # 

def lbck():

    import sys

    # get the first order tranverse order
    visite_sequence = first_order_sequence(pre,succ)

    # init parameters
    eta = computer_eta(workload,edge_num)
    eta_list = [eta for i in range(edge_num)]
    last_visited = 0
    computation_resouce_per_edge = [tmp for tmp in resouce_upbound]
    schedule_ans = [-1 for i in range(len(workload))]

    tmptaskindex = 0
    # 阶段1 获取分配策略
    while tmptaskindex < len(workload):
        if tmptaskindex < 0:
            return None
        tmptask = visite_sequence[tmptaskindex]
        tmptaskindex += 1
        match_edge = -1
        if last_visited == -1:
            # look for edge server where load is max the resource contraint is meet
            tmpetalist = [[i,tmp] for i,tmp in enumerate(eta_list)]
            tmpetalist.sort(key=lambda x: x[1], reverse=True)
            tmpmatch_edge = -1
            for i in range(len(tmpetalist)):
                if computation_resouce_per_edge[tmpetalist[i][0]] - workload[tmptask]/edge_computer_capability[tmpetalist[i][0]] >= 0:
                    tmpmatch_edge = tmpetalist[i][0]
                    break
            if tmpmatch_edge != -1:
                match_edge = tmpmatch_edge
            else:
                print("Can not find the edge server where meet the computation resouce upbound for task {0}".format(
                    tmptask
                ))
                return None

        else:
            if computation_resouce_per_edge[last_visited] - workload[tmptask]/edge_computer_capability[last_visited] >= 0:
                match_edge = last_visited
            else:
                print("Can not find the edge server where meet the computation resouce upbound for task {0}".format(
                    tmptask
                ))
                last_visited = -1
                tmptaskindex -= 1
                continue

        # reset the eta and last visited
        if workload[tmptask] <= eta_list[match_edge]:
            eta_list[match_edge] -= workload[tmptask]
            computation_resouce_per_edge[match_edge] -= workload[tmptask]/edge_computer_capability[match_edge]
            schedule_ans[tmptask] = match_edge

            last_visited = match_edge
        else:
            if workload[tmptask] > max(eta_list):
                last_visited = eta_list.index(max(eta_list))
                if computation_resouce_per_edge[last_visited] - workload[tmptask] / edge_computer_capability[
                    last_visited] >= 0:
                    match_edge = last_visited
                    eta_list[match_edge] -= workload[tmptask]
                    computation_resouce_per_edge[match_edge] -= workload[tmptask] / edge_computer_capability[match_edge]
                    schedule_ans[tmptask] = match_edge
                    continue
                else:
                    print("Can not find the balance way for the computation resouce constraint for task {0}".format(tmptask))
                    return None

            last_visited = -1
            tmptaskindex -= 1
            continue
    
    # 阶段2 reservation 调度

    # based on the list to computer the begin time and end time
    answer_list = [[] for i in range(edge_num)]
    aft_list = [-1 for i in range(len(workload))]
    ava_time_now = [[[0, sys.maxsize]] for i in range(len(ava_time_list))]

    for tmptask in visite_sequence:
        first_task = tmptask  # select the first task
        # compute the eft for each processor using the insertion-based scheduling policy
        i = schedule_ans[tmptask]
        ava_time_now[i].sort(key=lambda x: x[0], reverse=False)
        if len(pre[first_task]) == 0 or pre[first_task] == 0:
            est = 0
 
        else:
            est = max([ava_time_now[i][0][0],
                       max([aft_list[j] + computer_Cij(answer_list, j, first_task, i) for j in pre[first_task]])])

        eft = workload[first_task] / edge_computer_capability[i] + est

        # assign the task to the minimized eft task list
        minimize_processor = i
        task_best_time_intervel = [est, eft]
        match = False
        
        min_epoch_index = -1
        # 选择 k 
        # 处理分支 最佳时隙在两个可用时隙的中间 中间属于后一个epoch
        for j in range(len(decision_time_list[minimize_processor])):
            if j * default_timewindow <= est and eft <= (j+1) * default_timewindow :
                min_epoch_index = j
                break 
            if (j+1) * default_timewindow > est and (j+1) * default_timewindow <= eft :
                min_epoch_index = j + 1
                break 

        if min_epoch_index == -1:
            print("task {0} is out of time for offloading".format(tmptask))
            return -1
        
        # 判断 最佳时隙 是否 在private task interval间 
        # 是 推迟执行是否 违反 private task 超出 epoch时间约束
        # 是 违反约束 则将best_interval 调整到  private task 的开始
        # 不违反-则将private-task 调整 并将则将best_interval 按原来时间调度

        # for interval_index in range(min_epoch_index,decistion_time_number,1):
        private_task_interval = decision_time_list[minimize_processor][min_epoch_index]
        if not(min_epoch_index * default_timewindow <= task_best_time_intervel[0] and task_best_time_intervel[1] <= private_task_interval[0]) and not (private_task_interval[1] <= task_best_time_intervel[0] and task_best_time_intervel[1] <= (min_epoch_index+1) * default_timewindow):
            if task_best_time_intervel[1] + (private_task_interval[1] - private_task_interval[0]) > (min_epoch_index+1) * default_timewindow:
                task_best_time_intervel[0] = private_task_interval[1]
                task_best_time_intervel[1] = private_task_interval[1] + (eft - est)
            else:
                tmp = private_task_interval[1] - private_task_interval[0]
                decision_time_list[minimize_processor][min_epoch_index][0] = eft
                decision_time_list[minimize_processor][min_epoch_index][1] = eft + tmp
        # else:
        #     break 

        for i, tmp in enumerate(ava_time_now[minimize_processor]):
            start_time = -1
            end_time = -1

            # the best time intervel match case
            if tmp[0] <= task_best_time_intervel[0] and tmp[1] >= task_best_time_intervel[1]:  # case 1 为空
                match = True
                start_time = task_best_time_intervel[0] 
                end_time = task_best_time_intervel[1]   

            if tmp[0] <= task_best_time_intervel[0] and tmp[1] < task_best_time_intervel[1]:  # case 2
                pass
            if tmp[1] >= task_best_time_intervel[0]:  # case 3
                pass

            if tmp[0] >= task_best_time_intervel[0]:  # case 4 and case 5
                if workload[first_task] / edge_computer_capability[minimize_processor] <= tmp[1] - tmp[0]:
                    match = True
                    start_time = tmp[0]
                    end_time = tmp[0] + workload[first_task] / edge_computer_capability[minimize_processor]
                else:
                    pass

        

            if match:
                # change the avatime
                if tmp[0] == start_time:
                    ava_time_now[minimize_processor][i][0] = end_time
                elif tmp[1] == end_time:
                    ava_time_now[minimize_processor][i][1] = start_time

                if tmp[0] < start_time and tmp[1] > end_time:
                    tmpendtime = tmp[1]
                    ava_time_now[minimize_processor][i][0] = end_time
                    # ava_time_now[minimize_processor].append([end_time, tmpendtime])

                # add to the answer
                answer_list[minimize_processor].append([first_task, start_time, end_time])

                # change the finish time list
                aft_list[first_task] = end_time

                break

        if not match:
            print("Can not find a ava for task {0} from start time {1} to end time {2}".format(first_task,
                                                                                               task_best_time_intervel[
                                                                                                   0],
                                                                                               task_best_time_intervel[
                                                                                                   1]))
            return None
    return answer_list

def heft_time_reservation():
    import math
    import numpy as np
    from queue import Queue
    import sys

    global workload
    global datasize
    # print("decision_time_list")
    # print(decision_time_list)
    unavailable_time_list = [[] for i in range(edge_num)]
    for i,tmp_device_time in enumerate(decision_time_list):
        for tmp_time in tmp_device_time:
            unavailable_time_list[i].append(tmp_time.copy())
    
    # set computation cost and communication cost
    computation_cost = [np.average([tmpw/tmpC for tmpC in edge_computer_capability]) for tmpw in workload]

    communication_cost = np.array(datasize).copy()
    W_list = []
    for tmp in W:
        W_list += tmp
    for i in range(len(datasize)):
        for j in range(len(datasize[i])):
            communication_cost[i][j] = np.average([communication_cost[i][j]/tmp for tmp in W_list])

    # computer the rank upward for all traversing task graph upward starting from the exit task
    task_uprank = [0 for i in range(len(workload))]

    last_task = -1 # find the last task
    for i, tmp in enumerate(succ):
        if len(tmp) == 0:
            last_task = i
    tranverse_queue = Queue()
    visited = set()

    tranverse_queue.put(last_task)
    visited.add(last_task)
    # while not tranverse_queue.empty():
    #     tmp = tranverse_queue.get()
    #     visited.add(tmp)
    #
    #     if len(succ[tmp]) == 0:
    #         task_uprank[tmp] = computation_cost[tmp]
    #     else:
    #         task_uprank[tmp] = computation_cost[tmp] + np.max([communication_cost[tmp][j]+task_uprank[j] for j in succ[tmp]])
    #     for tmppre in pre[tmp]:
    #         if tmppre not in visited:
    #             tranverse_queue.put(tmppre)

    post_order_sequence = first_order_sequence(pre,succ)
    post_order_sequence.reverse()
    for tmp in post_order_sequence:
        if len(succ[tmp]) == 0:
            task_uprank[tmp] = computation_cost[tmp]
        else:
            task_uprank[tmp] = computation_cost[tmp] + np.max(
                [communication_cost[tmp][order2subtaskindex_map[j]] + task_uprank[j] for j in succ[tmp]])

    task_uprank = [[i,tmp] for i,tmp in enumerate(task_uprank)]
    task_schedule_sequence = [i for i,tmp in enumerate(task_uprank)]

    # sort the tasks scheduling list by no increaseing
    task_uprank.sort(reverse=True, key=takeSecond)

    # init the answer list
    answer_list = [[] for i in range(len(ava_time_list))]
    ava_time_now = [[[0, sys.maxsize]] for i in range(len(ava_time_list))] # copy the ava_time_list
    # for i in range(len(ava_time_list)):
    #     for tmp in ava_time_list[i]:
    #         ava_time_now[i].append(tmp.copy())
    #     ava_time_now[i].sort(key=takeSecond, reverse=False)

    # AFT list the actual finish time list
    aft_list = [-1 for i in range(len(workload))]


    while len(task_uprank) != 0:
        first_task = task_uprank[0][0] # select the first task

        eft_first_task_list = []
        # compute the eft for each processor using the insertion-based scheduling policy
        for i in range(len(ava_time_list)):
            ava_time_now[i].sort(key=lambda x : x[0], reverse=False)
            if len(pre[first_task]) == 0:
                est = 0
            elif pre[first_task][0]==0:
                est = ava_time_now[i][0][0]
            else:
                est = max([ava_time_now[i][0][0], max([aft_list[j] + computer_Cij(answer_list, j, first_task, i) for j in pre[first_task]])])

            eft = workload[first_task] / edge_computer_capability[i] + est

            eft_first_task_list.append([i, est, eft])

        # sor the list and choose the minimize eft of tasks
        eft_list = eft_first_task_list
        eft_list.sort(key=lambda x: x[2], reverse=False)

        # assign the task to the minimized eft task list
        minimize_processor = eft_list[0][0]
        task_best_time_intervel = [eft_list[0][1], eft_list[0][2]]
        match = False
        for i,tmp in enumerate(ava_time_now[minimize_processor]):
            start_time = -1
            end_time = -1

            # the best time intervel match case
            if tmp[0] <= task_best_time_intervel[0] and tmp[1] >= task_best_time_intervel[1]: # case 1
                match = True
                start_time = task_best_time_intervel[0]
                end_time = task_best_time_intervel[1]

            if tmp[0] <= task_best_time_intervel[0] and tmp[1] < task_best_time_intervel[1]: # case 2
                pass
            if tmp[1] >= task_best_time_intervel[0]: # case 3
                pass

            if tmp[0] >= task_best_time_intervel[0]: # case 4 and case 5
                if workload[first_task]/edge_computer_capability[minimize_processor] <= tmp[1]-tmp[0]:
                    match = True
                    start_time = tmp[0]
                    end_time = tmp[0] + workload[first_task]/edge_computer_capability[minimize_processor]
                else:
                    pass

            if match:
                # change the avatime
                if tmp[0] == start_time:
                    ava_time_now[minimize_processor][i][0] = end_time
                elif tmp[1] == end_time:
                    ava_time_now[minimize_processor][i][1] = start_time

                if tmp[0] < start_time and tmp[1] > end_time:
                    # tmpendtime = tmp[1]
                    ava_time_now[minimize_processor][i][0] = end_time
                    # ava_time_now[minimize_processor].append([end_time, tmpendtime])

                # add to the answer
                answer_list[minimize_processor].append([first_task, start_time, end_time])

                # change the finish time list
                aft_list[first_task] = end_time

                break

        if not match:
            print("Can not find a ava for task {0} from start time {1} to end time {2}".format(first_task,
                                                                                               task_best_time_intervel[0],
                                                                                               task_best_time_intervel[1]))
            return None
        task_uprank.remove(task_uprank[0])

    # reset the anser list based on the ava time given
    ava_time_intervals = [[] for i in range(len(ava_time_list))]  # copy the ava_time_list
    for i in range(len(ava_time_list)):
        for epoch_k in range(round(len(ava_time_list[i])/2)):
            tmp_interval = []
            tmp_interval.append(ava_time_list[i][epoch_k*2].copy())
            ava_time_intervals[i].append(tmp_interval)

    ans_dict = {}
    for i in range(len(answer_list)):
        for j in range(len(answer_list[i])):
            ans_dict[answer_list[i][j][0]] = [i, answer_list[i][j][1], answer_list[i][j][2]]

    aft_list = [0 for i in range(len(workload))]

    for tmptask in task_schedule_sequence:
        first_task = tmptask
        minimize_processor = ans_dict[first_task][0]
        match = False
        # reset the best time intervel
        if len(pre[first_task]) == 0 or pre[first_task][0]==0:
            est = ans_dict[first_task][1]
        else:
            est = max([aft_list[j] + computer_Cij(answer_list, j, first_task, minimize_processor)
                       for j in pre[first_task]])

        eft = workload[first_task] / edge_computer_capability[minimize_processor] + est

        task_best_time_intervel = [est, eft]

        min_epoch_index = -1

        # 选择 k 
        # 处理分支 最佳时隙在两个可用时隙的中间 中间属于后一个epoch
        for j in range(len(unavailable_time_list[minimize_processor])):
            if j * default_timewindow <= task_best_time_intervel[0] and task_best_time_intervel[1] <= (j+1) * default_timewindow :
                min_epoch_index = j
                break 
            if task_best_time_intervel[0] <= (j+1) * default_timewindow  and (j+1) * default_timewindow < task_best_time_intervel[1] :
                min_epoch_index = j + 1
                break
        

        if min_epoch_index == -1 or min_epoch_index>=len(unavailable_time_list[minimize_processor]):
            print("task {0} is out of time for offloading".format(tmptask))
            return -1

        epoch_number = len(unavailable_time_list[minimize_processor])
        for min_epoch_index in range(min_epoch_index,epoch_number,1):
            private_task_interval = unavailable_time_list[minimize_processor][min_epoch_index]
            ava_time_intervals[minimize_processor][min_epoch_index].sort(key=lambda  x: x[0], reverse=False)
            available_number = len(ava_time_intervals[minimize_processor][min_epoch_index])
            for i, tmp in enumerate(ava_time_intervals[minimize_processor][min_epoch_index]):
                start_time = -1
                end_time = -1
                if i != available_number-1: # 不是最后一个间隙
                    # the best time intervel match case
                    if tmp[0] <= task_best_time_intervel[0] and tmp[1] >= task_best_time_intervel[1]:  # case 1
                        match = True
                        start_time = task_best_time_intervel[0]
                        end_time = task_best_time_intervel[1]

                    if tmp[0] <= task_best_time_intervel[0] and tmp[1] < task_best_time_intervel[1]:  # case 2
                        pass
                    if tmp[1] >= task_best_time_intervel[0]:  # case 3
                        pass
                    
                    if tmp[0] >= task_best_time_intervel[0]:  # case 4 and case 5 卡在available time interval 前面
                        if workload[first_task] / edge_computer_capability[minimize_processor] <= tmp[1] - tmp[0]:
                            match = True
                            start_time = tmp[0]
                            end_time = tmp[0] + workload[first_task] / edge_computer_capability[minimize_processor]
                        else:
                            continue
                else:
                    # the best time intervel match case
                    if tmp[0] <= task_best_time_intervel[0] and tmp[1] >= task_best_time_intervel[1]:  # case 1
                        match = True
                        start_time = task_best_time_intervel[0]
                        end_time = task_best_time_intervel[1]

                    if tmp[0] <= task_best_time_intervel[0] and tmp[1] < task_best_time_intervel[1]:  # case 2
                        unava_tmp = private_task_interval[1] - private_task_interval[0]
                        ava_tmp = task_best_time_intervel[1] - task_best_time_intervel[0]
                        ava_tmp_end = private_task_interval[0] + ava_tmp
                        if ava_tmp_end + unava_tmp <= (min_epoch_index+1) * default_timewindow:
                            if tmp[1] == private_task_interval[0]:
                                # 增长
                                match = True
                                start_time = tmp[0]
                                end_time = start_time + ava_tmp
                                ava_time_intervals[minimize_processor][min_epoch_index][-1][1] = end_time
                                unavailable_time_list[minimize_processor][min_epoch_index][0] = end_time
                                unavailable_time_list[minimize_processor][min_epoch_index][1] = end_time + unava_tmp
                            else:
                                # 扩展
                                match = True
                                ava_time_intervals[minimize_processor][min_epoch_index].append([private_task_interval[0],ava_tmp_end])
                                start_time = private_task_interval[0]
                                end_time = ava_tmp_end
                                unavailable_time_list[minimize_processor][min_epoch_index][0] = end_time
                                unavailable_time_list[minimize_processor][min_epoch_index][1] = end_time + unava_tmp
                        else:
                            continue
                    if tmp[1] >= task_best_time_intervel[0]:  # case 3
                        pass
                    
                    if tmp[0] > task_best_time_intervel[0]:  # case 4 and case 5 卡在available time interval 前面
                        if workload[first_task] / edge_computer_capability[minimize_processor] <= tmp[1] - tmp[0]:
                            match = True
                            start_time = tmp[0]
                            end_time = tmp[0] + workload[first_task] / edge_computer_capability[minimize_processor]
                        else:
                            unava_tmp = private_task_interval[1] - private_task_interval[0]
                            ava_tmp = task_best_time_intervel[1] - task_best_time_intervel[0]
                            ava_tmp_end = tmp[0] + ava_tmp
                            if ava_tmp_end + unava_tmp <= (min_epoch_index+1) * default_timewindow:
                                if tmp[1] == private_task_interval[0]:
                                    # 增长
                                    match = True
                                    start_time = tmp[0]
                                    end_time = start_time + ava_tmp
                                    ava_time_intervals[minimize_processor][min_epoch_index][-1][1] = end_time
                                    unavailable_time_list[minimize_processor][min_epoch_index][0] = end_time
                                    unavailable_time_list[minimize_processor][min_epoch_index][1] = end_time + unava_tmp
                                else:
                                    # 扩展
                                    match = True
                                    ava_time_intervals[minimize_processor][min_epoch_index].append([private_task_interval[0],ava_tmp_end])
                                    start_time = private_task_interval[0]
                                    end_time = ava_tmp_end
                                    unavailable_time_list[minimize_processor][min_epoch_index][0] = end_time
                                    unavailable_time_list[minimize_processor][min_epoch_index][1] = end_time + unava_tmp
                            else:
                                continue
                if match:
                    # change the avatime
                    if tmp[0] == start_time:
                        ava_time_intervals[minimize_processor][min_epoch_index][i][0] = end_time
                        # ava_time_now[minimize_processor][0][0] = end_time
                    elif tmp[1] == end_time:
                        ava_time_intervals[minimize_processor][min_epoch_index][i][1] = start_time

                    if tmp[0] < start_time and tmp[1] > end_time:
                        tmpendtime = tmp[1]
                        ava_time_intervals[minimize_processor][min_epoch_index][i][1] = start_time
                        ava_time_intervals[minimize_processor][min_epoch_index].append([end_time, tmpendtime])
                        ava_time_intervals[minimize_processor][min_epoch_index].sort(key=lambda  x: x[0], reverse=False)
                        ava_time_now[minimize_processor][0][0] = end_time
                    # add to the answer
                    # answer_list[k][j][1] = start_time
                    # answer_list[k][j][2] = end_time

                    ans_dict[first_task][1] = start_time
                    ans_dict[first_task][2] = end_time
                    aft_list[first_task] = end_time
                    break
            
            if match:
                break
            else:
                ava_tmp = task_best_time_intervel[1] - task_best_time_intervel[0]
                task_best_time_intervel[0] = (min_epoch_index+1) * default_timewindow
                task_best_time_intervel[1] = task_best_time_intervel[0] + ava_tmp

        if not match:
            print("Can not find a ava for task {0} from start time {1} to end time {2}".format(first_task,
                                                                                            task_best_time_intervel[
                                                                                                0],
                                                                                            task_best_time_intervel[
                                                                                                1]))
            return None
            
    # change the ans dict to ans list
    actual_ans_list = [[] for i in range(len(ava_time_list))]
    for tmptask in ans_dict:
        actual_ans_list[ans_dict[tmptask][0]].append([tmptask, ans_dict[tmptask][1], ans_dict[tmptask][2]])

    # return answer_list, theory_answer_list
    return actual_ans_list

# NSGA-!!
def selection(individual_list):
    #
    import numpy as np
    evaluate_list = []
    ans_dict_list = []
    ans_list = []
    for i in range(len(individual_list)):
        ans,ans_dict = evaluate(individual_list[i])
        fitness = get_max_time(ans)
        ans_dict_list.append(ans_dict)
        evaluate_list.append(fitness)
        ans_list.append(ans)
    
    # selection_number = round(top_ratio * len(individual_list))
    selection_individual = np.argsort(np.array(evaluate_list))[:2]
    best_index = selection_individual[0]

    return selection_individual,evaluate_list[best_index],ans_dict_list[best_index],ans_list[best_index]

def generate_offspring(individual_list,parents_list,population,crosscover_ratio,mutation_ratio):
    offsprings_list = []
    while len(offsprings_list) < population:
        parent1 = individual_list[parents_list[0]]
        parent2 = individual_list[parents_list[1]]

        offspring1 = individual_list[parents_list[0]].copy()
        offspring2 = individual_list[parents_list[1]].copy()

        gene_number = len(parent1)
        import random
        for i in range(gene_number):
            if random.random() <= crosscover_ratio:
                offspring1[i][2] = parent2[i][2]
                offspring2[i][2] = parent1[i][2]
            if random.random() <= mutation_ratio:

                alternative1 = [i for i in range(edge_num)]
                alternative1.remove(offspring1[i][2])
                alternative2 = [i for i in range(edge_num)]
                alternative2.remove(offspring2[i][2])
                offspring1[i][2] = random.choice(alternative1)
                offspring2[i][2] = random.choice(alternative2)
        offsprings_list.append(offspring1)
        offsprings_list.append(offspring2)
    return offsprings_list

def chromosome_code(visist_order):
    # init coding
    import random
    subtask_nums = len(workload)
    chromosome_genes_list = []

    for tmptask_index in visist_order:
        subtask_index = order2subtaskindex_map[tmptask_index]
        task_index = order2taskindex_map[tmptask_index]

        server_index = random.sample([i for i in range(edge_num)],1)[0]

        chromosome_genes_list.append([task_index,subtask_index,server_index])
    return chromosome_genes_list

def evaluate(individual):
    # according to visit order(individual 构造的时候已经被隐含了visit的顺序) to offload subtask and calculate the completion time in a individual
    ava_time_now_list = [0 for i in range(edge_num)]
    answear_list = [[] for i in range(edge_num)]
    ans_dict = {}
    for task_offload_deicision in individual:
        task_index = task_offload_deicision[0]
        subtask_index = task_offload_deicision[1]
        server_index = task_offload_deicision[2]
        if task_index == -1:
            # 虚拟节点 跳过调度
            continue 
        tmptask =taskindex2order_map[task_index][subtask_index]
        execution_time = workload[tmptask]/edge_computer_capability[server_index]

        if len(pre[tmptask]) == 0 or pre[tmptask][0] == 0:
            ready_time = 0
        else:
            ready_time = max([computer_Cij(answear_list,j,tmptask,server_index) for j in pre[tmptask]])
        
        ava_time_now = ava_time_now_list[server_index]
        start_time = max([ready_time,ava_time_now])
        end_time = start_time + execution_time

        answear_list[server_index].append([tmptask,start_time,end_time])
        ans_dict[tmptask] = [server_index,start_time,end_time]
        ava_time_now_list[server_index] = end_time

    return answear_list,ans_dict

def NSGA():
    # main
    import sys
    population = 10
    visit_order = first_order_sequence(pre,succ)
    individual_list = []
    top_ratio = 0.1
    crosscover_ratio = 0.65
    mutation_ratio = 0.1
    iter_size = 2
    iter_number = 0
    best_individual = None 
    best_ans_dict = None
    best_fitness = float("inf")
    for i in range(population):
        individual_list.append(chromosome_code(visit_order))

    while iter_number < iter_size:
        parents_list,local_fitness,local_ans_dict,local_answear = selection(individual_list)
        offsprings_list = generate_offspring(individual_list,parents_list,population,crosscover_ratio,mutation_ratio)
        
        if best_fitness > local_fitness:
            best_individual_alternative = individual_list[parents_list[0]]
            best_ans_dict = local_ans_dict
            best_fitness = local_fitness
            nsga_answear = local_answear
        iter_number += 1
        individual_list = offsprings_list

    ava_time_now = [[] for i in range(len(ava_time_list))]  # copy the ava_time_list

    for i in range(len(ava_time_list)):
        for tmp in ava_time_list[i]:
            ava_time_now[i].append(tmp.copy())
        ava_time_now[i].sort(key=takeSecond, reverse=False)
    
    aft_list = [0 for i in range(len(workload))]

    for tmptask in visit_order:
        if tmptask == 0 or tmptask == len(visit_order)-1 :
            #虚拟节点跳过
            continue
        first_task = tmptask
        minimize_processor = best_ans_dict[tmptask][0]
        match = False
        # reset the best time intervel
        if len(pre[first_task]) == 0 or pre[first_task][0]==0:
            est = best_ans_dict[first_task][1]
        else:
            est = max([aft_list[j] + computer_Cij(nsga_answear, j, first_task, minimize_processor)
                       for j in pre[first_task]])

        eft = workload[first_task] / edge_computer_capability[minimize_processor] + est

        task_best_time_intervel = [est, eft]

        ava_time_now[minimize_processor].sort(key=lambda  x: x[0], reverse=False)
        for i, tmp in enumerate(ava_time_now[minimize_processor]):
            start_time = -1
            end_time = -1

            # the best time intervel match case
            if tmp[0] <= task_best_time_intervel[0] and tmp[1] >= task_best_time_intervel[1]:  # case 1
                match = True
                start_time = task_best_time_intervel[0]
                end_time = task_best_time_intervel[1]

            if tmp[0] <= task_best_time_intervel[0] and tmp[1] < task_best_time_intervel[1]:  # case 2
                pass
            if tmp[1] >= task_best_time_intervel[0]:  # case 3
                pass

            if tmp[0] >= task_best_time_intervel[0]:  # case 4 and case 5
                if workload[first_task] / edge_computer_capability[minimize_processor] <= tmp[1] - tmp[0]:
                    match = True
                    start_time = tmp[0]
                    end_time = tmp[0] + workload[first_task] / edge_computer_capability[minimize_processor]
                else:
                    pass

            if match:
                # change the avatime
                if tmp[0] == start_time:
                    ava_time_now[minimize_processor][i][0] = end_time
                elif tmp[1] == end_time:
                    ava_time_now[minimize_processor][i][1] = start_time

                if tmp[0] < start_time and tmp[1] > end_time:
                    tmpendtime = tmp[1]
                    ava_time_now[minimize_processor][i][1] = start_time
                    ava_time_now[minimize_processor].append([end_time, tmpendtime])

                best_ans_dict[first_task][1] = start_time
                best_ans_dict[first_task][2] = end_time
                aft_list[first_task] = end_time
                break

        if not match:
            print("Can not find a ava for task {0} from start time {1} to end time {2}".format(first_task,
                                                                                               task_best_time_intervel[
                                                                                                   0],
                                                                                               task_best_time_intervel[
                                                                                                  1]))
            return None

    # change the ans dict to ans list
    nsga_actual_ans_list = [[] for i in range(len(ava_time_list))]
    for tmptask in best_ans_dict:
        if tmptask == 0 or tmptask == len(visit_order)-1:
            continue
        nsga_actual_ans_list[best_ans_dict[tmptask][0]].append([tmptask, best_ans_dict[tmptask][1], best_ans_dict[tmptask][2]])


    return nsga_actual_ans_list

'heft 算法'
def heft():
    import math
    import numpy as np
    from queue import Queue
    import sys

    # set computation cost and communication cost
    computation_cost = [np.average([tmpw/tmpC for tmpC in edge_computer_capability]) for tmpw in workload]

    communication_cost = np.array(datasize).copy()
    W_list = []
    for tmp in W:
        W_list += tmp
    for i in range(len(datasize)):
        for j in range(len(datasize[i])):
            communication_cost[i][j] = np.average([communication_cost[i][j]/tmp for tmp in W_list])

    # computer the rank upward for all traversing task graph upward starting from the exit task
    task_uprank = [0 for i in range(len(workload))]

    last_task = -1 # find the last task
    for i, tmp in enumerate(succ):
        if len(tmp) == 0:
            last_task = i
    tranverse_queue = Queue()
    visited = set()

    tranverse_queue.put(last_task)
    visited.add(last_task)

    post_order_sequence = first_order_sequence(pre,succ)
    post_order_sequence.reverse()
    for tmp in post_order_sequence:
        if len(succ[tmp]) == 0:
            task_uprank[tmp] = computation_cost[tmp]
        else:
            task_uprank[tmp] = computation_cost[tmp] + np.max(
                [communication_cost[tmp][order2subtaskindex_map[j]] + task_uprank[j] for j in succ[tmp]])

    task_uprank = [[i,tmp] for i,tmp in enumerate(task_uprank)]
    task_schedule_sequence = [i for i,tmp in enumerate(task_uprank)]

    # sort the tasks scheduling list by no increaseing
    task_uprank.sort(reverse=True, key=takeSecond)

    # init the answer list
    answer_list = [[] for i in range(len(ava_time_list))]
    ava_time_now = [[[0, sys.maxsize]] for i in range(len(ava_time_list))] # copy the ava_time_list
    # for i in range(len(ava_time_list)):
    #     for tmp in ava_time_list[i]:
    #         ava_time_now[i].append(tmp.copy())
    #     ava_time_now[i].sort(key=takeSecond, reverse=False)

    # AFT list the actual finish time list
    aft_list = [-1 for i in range(len(workload))]


    while len(task_uprank) != 0:
        first_task = task_uprank[0][0] # select the first task

        eft_first_task_list = []
        # compute the eft for each processor using the insertion-based scheduling policy
        for i in range(len(ava_time_list)):
            ava_time_now[i].sort(key=lambda x : x[0], reverse=False)
            if len(pre[first_task]) == 0:
                est = 0
            elif pre[first_task][0]==0:
                est = ava_time_now[i][0][0]
            else:
                est = max([ava_time_now[i][0][0], max([aft_list[j] + computer_Cij(answer_list, j, first_task, i) for j in pre[first_task]])])

            eft = workload[first_task] / edge_computer_capability[i] + est

            eft_first_task_list.append([i, est, eft])

        # sor the list and choose the minimize eft of tasks
        eft_list = eft_first_task_list
        eft_list.sort(key=lambda x: x[2], reverse=False)

        # assign the task to the minimized eft task list
        minimize_processor = eft_list[0][0]
        task_best_time_intervel = [eft_list[0][1], eft_list[0][2]]
        match = False
        for i,tmp in enumerate(ava_time_now[minimize_processor]):
            start_time = -1
            end_time = -1 

            # the best time intervel match case
            if tmp[0] <= task_best_time_intervel[0] and tmp[1] >= task_best_time_intervel[1]: # case 1
                match = True
                start_time = task_best_time_intervel[0]
                end_time = task_best_time_intervel[1]

            if tmp[0] <= task_best_time_intervel[0] and tmp[1] < task_best_time_intervel[1]: # case 2
                pass
            if tmp[1] >= task_best_time_intervel[0]: # case 3
                pass

            if tmp[0] >= task_best_time_intervel[0]: # case 4 and case 5
                if workload[first_task]/edge_computer_capability[minimize_processor] <= tmp[1]-tmp[0]:
                    match = True
                    start_time = tmp[0]
                    end_time = tmp[0] + workload[first_task]/edge_computer_capability[minimize_processor]
                else:
                    pass

            if match:
                # change the avatime
                if tmp[0] == start_time:
                    ava_time_now[minimize_processor][i][0] = end_time
                elif tmp[1] == end_time:
                    ava_time_now[minimize_processor][i][1] = start_time

                if tmp[0] < start_time and tmp[1] > end_time:
                    tmpendtime = tmp[1]
                    ava_time_now[minimize_processor][i][0] = end_time
                    # ava_time_now[minimize_processor].append([end_time, tmpendtime])

                # add to the answer
                answer_list[minimize_processor].append([first_task, start_time, end_time])

                # change the finish time list
                aft_list[first_task] = end_time

                break

        if not match:
            print("Can not find a ava for task {0} from start time {1} to end time {2}".format(first_task,
                                                                                               task_best_time_intervel[0],
                                                                                               task_best_time_intervel[1]))
            return None
        task_uprank.remove(task_uprank[0])

    # reset the anser list based on the ava time given
    ava_time_now = [[] for i in range(len(ava_time_list))]  # copy the ava_time_list
    for i in range(len(ava_time_list)):
        for tmp in ava_time_list[i]:
            ava_time_now[i].append(tmp.copy())
        ava_time_now[i].sort(key=takeSecond, reverse=False)
    # theory_answer_list = [[] for i in range(len(ava_time_list))]
    # for i in range(len(answer_list)):
    #     for j in range(len(answer_list[i])):
    #         theory_answer_list[i].append(answer_list[i][j].copy())

    ans_dict = {}
    for i in range(len(answer_list)):
        for j in range(len(answer_list[i])):
            ans_dict[answer_list[i][j][0]] = [i, answer_list[i][j][1], answer_list[i][j][2]]

    aft_list = [0 for i in range(len(workload))]
    
    visited_order = first_order_sequence(pre,succ)
    for tmptask in visited_order:
        first_task = tmptask
        minimize_processor = ans_dict[first_task][0]
        match = False
        # reset the best time intervel
        if len(pre[first_task]) == 0 or pre[first_task][0]==0:
            est = ans_dict[first_task][1]
        else:
            est = max([aft_list[j] + computer_Cij(answer_list, j, first_task, minimize_processor)
                       for j in pre[first_task]])

        eft = workload[first_task] / edge_computer_capability[minimize_processor] + est

        task_best_time_intervel = [est, eft]

        ava_time_now[minimize_processor].sort(key=lambda  x: x[0], reverse=False)
        for i, tmp in enumerate(ava_time_now[minimize_processor]):
            start_time = -1
            end_time = -1

            # the best time intervel match case
            if tmp[0] <= task_best_time_intervel[0] and tmp[1] >= task_best_time_intervel[1]:  # case 1
                match = True
                start_time = task_best_time_intervel[0]
                end_time = task_best_time_intervel[1]

            if tmp[0] <= task_best_time_intervel[0] and tmp[1] < task_best_time_intervel[1]:  # case 2
                pass
            if tmp[1] >= task_best_time_intervel[0]:  # case 3
                pass

            if tmp[0] >= task_best_time_intervel[0]:  # case 4 and case 5
                if workload[first_task] / edge_computer_capability[minimize_processor] <= tmp[1] - tmp[0]:
                    match = True
                    start_time = tmp[0]
                    end_time = tmp[0] + workload[first_task] / edge_computer_capability[minimize_processor]
                else:
                    pass

            if match:
                # change the avatime
                if tmp[0] == start_time:
                    ava_time_now[minimize_processor][i][0] = end_time
                elif tmp[1] == end_time:
                    ava_time_now[minimize_processor][i][1] = start_time

                if tmp[0] < start_time and tmp[1] > end_time:
                    tmpendtime = tmp[1]
                    ava_time_now[minimize_processor][i][1] = start_time
                    ava_time_now[minimize_processor].append([end_time, tmpendtime])

                # add to the answer
                # answer_list[k][j][1] = start_time
                # answer_list[k][j][2] = end_time

                ans_dict[first_task][1] = start_time
                ans_dict[first_task][2] = end_time
                aft_list[first_task] = end_time
                break

        if not match:
            print("Can not find a ava for task {0} from start time {1} to end time {2}".format(first_task,
                                                                                               task_best_time_intervel[
                                                                                                   0],
                                                                                               task_best_time_intervel[
                                                                                                  1]))
            return None

    # change the ans dict to ans list
    actual_ans_list = [[] for i in range(len(ava_time_list))]
    for tmptask in ans_dict:
        actual_ans_list[ans_dict[tmptask][0]].append([tmptask, ans_dict[tmptask][1], ans_dict[tmptask][2]])

    # return answer_list, theory_answer_list
    return actual_ans_list


'greedy algorithm'
def greedy():
    # get visited sequence
    visited_sequence = first_order_sequence(pre,succ)

    # init paramerters
    answer_list = [[] for i in range(len(ava_time_list))]
    aft_list = [0 for i in range(len(workload))]
    ava_time_now = [[] for i in range(len(ava_time_list))]  # copy the ava_time_list
    for i in range(len(ava_time_list)):
        for tmp in ava_time_list[i]:
            ava_time_now[i].append(tmp.copy())
        ava_time_now[i].sort(key=takeSecond, reverse=False)

    for tmptask in visited_sequence:
        first_task = tmptask  # select the first task

        eft_first_task_list = []
        # compute the eft for each processor using the insertion-based scheduling policy
        for i in range(len(ava_time_list)):
            if len(ava_time_now[i]) == 0:
                continue

            ava_time_now[i].sort(key=lambda x: x[0], reverse=False)

            ready_time = 0
            if len(pre[first_task]) == 0 or pre[first_task][0] == 0:
                ready_time = 0
            else:
                ready_time = max([aft_list[j] + computer_Cij(answer_list, j, first_task, i) for j in pre[first_task]])
            early_ava_time = -1
            for tmpavatime in ava_time_now[i]:
                if tmpavatime[1] - ready_time >= workload[first_task] / edge_computer_capability[i]:
                    if tmpavatime[0] + workload[first_task] / edge_computer_capability[i] <= tmpavatime[1]:
                        early_ava_time = tmpavatime[0]
                        break
            # if len(pre[first_task]) == 0:
            #     est = 0
            # else:
            if early_ava_time != -1:
                est = max([early_ava_time, ready_time])

                eft = workload[first_task] / edge_computer_capability[i] + est

                eft_first_task_list.append([i, est, eft])

        # sor the list and choose the minimize eft of tasks
        eft_list = eft_first_task_list
        eft_list.sort(key=lambda x: x[2], reverse=False)

        # assign the task to the minimized eft task list
        minimize_processor = eft_list[0][0]
        task_best_time_intervel = [eft_list[0][1], eft_list[0][2]]
        match = False

        ava_time_now[minimize_processor].sort(key=lambda x: x[0], reverse=False)
        for i, tmp in enumerate(ava_time_now[minimize_processor]):
            start_time = -1
            end_time = -1

            # the best time intervel match case
            if tmp[0] <= task_best_time_intervel[0] and tmp[1] >= task_best_time_intervel[1]:  # case 1
                match = True
                start_time = task_best_time_intervel[0]
                end_time = task_best_time_intervel[1]

            if tmp[0] <= task_best_time_intervel[0] and tmp[1] < task_best_time_intervel[1]:  # case 2
                pass
            if tmp[1] >= task_best_time_intervel[0]:  # case 3
                pass

            if tmp[0] > task_best_time_intervel[0]:  # case 4 and case 5
                if workload[first_task] / edge_computer_capability[minimize_processor] <= tmp[1] - tmp[0]:
                    match = True
                    start_time = tmp[0]
                    end_time = tmp[0] + workload[first_task] / edge_computer_capability[minimize_processor]
                else:
                    pass

            if match:
                # change the avatime
                if tmp[0] == start_time:
                    ava_time_now[minimize_processor][i][0] = end_time
                elif tmp[1] == end_time:
                    ava_time_now[minimize_processor][i][1] = start_time

                if tmp[0] < start_time and tmp[1] > end_time:
                    tmpendtime = tmp[1]
                    ava_time_now[minimize_processor][i][1] = start_time
                    ava_time_now[minimize_processor].append([end_time, tmpendtime])

                # add to the answer
                answer_list[minimize_processor].append([first_task, start_time, end_time])

                aft_list[first_task] = end_time
                break

        if not match:
            print("Can not find a ava for task {0} from start time {1} to end time {2}".format(first_task,
                                                                                               task_best_time_intervel[
                                                                                                   0],
                                                                                               task_best_time_intervel[
                                                                                                   1]))
            return None

    return answer_list

'greedy algorithm'
def greedy_time_reservation():

    # get visited sequence
    visited_sequence = first_order_sequence(pre,succ)

    # init paramerters
    answer_list = [[] for i in range(len(ava_time_list))]
    aft_list = [0 for i in range(len(workload))]

    ava_time_now = reservation_reallocate_avatime(ava_time_list,decision_time_list)

    for tmptask in visited_sequence:
        first_task = tmptask  # select the first task
        eft_first_task_list = []
        # compute the eft for each processor using the insertion-based scheduling policy
        for i in range(len(ava_time_list)):
            if len(ava_time_now[i]) == 0:
                continue
            ava_time_now[i].sort(key=lambda x: x[0], reverse=False)

            ready_time = 0
            if len(pre[first_task]) == 0 or pre[first_task][0] == 0:
                ready_time = 0
            else:
                ready_time = max([aft_list[j] + computer_Cij(answer_list, j, first_task, i) for j in pre[first_task]])
            early_ava_time = -1
            execution_time = workload[first_task] / edge_computer_capability[i]
            for tmpavatime in ava_time_now[i]:
                if tmpavatime[1] - ready_time >= execution_time:
                    if tmpavatime[0] + execution_time <= tmpavatime[1]:
                        early_ava_time = tmpavatime[0]
                        break
            # if len(pre[first_task]) == 0:
            #     est = 0
            # else:
            if early_ava_time != -1:
                est = max([early_ava_time, ready_time])

                eft = workload[first_task] / edge_computer_capability[i] + est

                eft_first_task_list.append([i, est, eft])

        # sor the list and choose the minimize eft of tasks
        eft_list = eft_first_task_list
        eft_list.sort(key=lambda x: x[2], reverse=False)

        # assign the task to the minimized eft task list
        minimize_processor = eft_list[0][0]
        task_best_time_intervel = [eft_list[0][1], eft_list[0][2]]
        match = False

        ava_time_now[minimize_processor].sort(key=lambda x: x[0], reverse=False)

        for i, tmp in enumerate(ava_time_now[minimize_processor]):


            start_time = -1
            end_time = -1

            # the best time intervel match case
            if tmp[0] <= task_best_time_intervel[0] and tmp[1] >= task_best_time_intervel[1]:  # case 1
                match = True
                start_time = task_best_time_intervel[0]
                end_time = task_best_time_intervel[1]

            if tmp[0] <= task_best_time_intervel[0] and tmp[1] < task_best_time_intervel[1]:  # case 2
                pass
            if tmp[1] >= task_best_time_intervel[0]:  # case 3
                pass

            if tmp[0] > task_best_time_intervel[0]:  # case 4 and case 5
                if workload[first_task] / edge_computer_capability[minimize_processor] <= tmp[1] - tmp[0]:
                    match = True
                    start_time = tmp[0]
                    end_time = tmp[0] + workload[first_task] / edge_computer_capability[minimize_processor]
                else:
                    pass

            if match:
                # change the avatime
                if tmp[0] == start_time:
                    ava_time_now[minimize_processor][i][0] = end_time
                elif tmp[1] == end_time:
                    ava_time_now[minimize_processor][i][1] = start_time

                if tmp[0] < start_time and tmp[1] > end_time:
                    tmpendtime = tmp[1]
                    ava_time_now[minimize_processor][i][1] = start_time
                    ava_time_now[minimize_processor].append([end_time, tmpendtime])

                # add to the answer
                answer_list[minimize_processor].append([first_task, start_time, end_time])

                aft_list[first_task] = end_time
                break

        if not match:
            print("Can not find a ava for task {0} from start time {1} to end time {2}".format(first_task,
                                                                                               task_best_time_intervel[
                                                                                                   0],
                                                                                               task_best_time_intervel[
                                                                                                   1]))
            return None

    return answer_list

# online
'N look back adjust'
def nlook_back(n):

    # based on heft, get a anser list
    heft_ans = heft_offline()

    # copy the ava_time_list
    ava_time_nlookback = [[] for i in range(len(ava_time_list))]
    for i in range(len(ava_time_list)):
        for j in range(len(ava_time_list[i])):
            ava_time_nlookback[i].append(ava_time_list[i][j].copy())

    # init parameters
    running_task = [] # has running task in the look back time
    visited_order = first_order_sequence() # the first order visited orderf
    aft_list = [-1 for i in range(len(workload))] # the actual finish time for each task
    answer_list = [[] for i in range(len(ava_time_list))] # the total answer list

    # for 0 ~ n, get
    lookback_start_time = 0
    lookback_end_time = n
    while len(running_task) != len(workload):
        # reset time interval
        local_ava_time = [[] for i in range(len(ava_time_list))]
        for i in range(len(ava_time_list)):
            for tmpavatime in ava_time_nlookback[i]:
                if tmpavatime[1] <= lookback_end_time and tmpavatime[0] >= lookback_start_time:
                    local_ava_time[i].append(tmpavatime.copy())
                else:
                    if tmpavatime[0] < lookback_end_time and tmpavatime[0] >= lookback_start_time:
                        if tmpavatime[0] - lookback_end_time != 0:
                            local_ava_time[i].append([tmpavatime[0], lookback_end_time])

                    if tmpavatime[0] < lookback_start_time:
                        if tmpavatime[1] > lookback_start_time and tmpavatime[1] <= lookback_end_time:
                            if tmpavatime[0] - lookback_end_time != 0:
                                local_ava_time[i].append([lookback_start_time, tmpavatime[1]])


        # local running task
        local_running_task = []
        # local running answer
        local_running_answers = []
        # local task finish time
        local_aft_list = [tmp for tmp in aft_list]

        # heft answer list init with copy with answer list
        heft_local_answer_list = [[] for i in range(len(ava_time_list))]
        for i in range(len(answer_list)):
            for tmptimeinterval in answer_list[i]:
                heft_local_answer_list[i].append(tmptimeinterval.copy())

        # choose the tasks from unfinish task and get heft offline answer, task finish time
        for tmptask in visited_order:
            if tmptask not in running_task and tmptask not in local_running_task: # chooice the task from the unfinish task
                # get offline answer
                tmptaskans = get_offline_answer(heft_ans, tmptask)

                # if all the pre task has been done
                pre_done = True
                for tmppretask in pre[tmptask]:
                    if tmppretask not in running_task and tmppretask not in local_running_task:
                        pre_done = False

                if pre_done:
                    if len(pre[tmptask]) == 0:
                        est = tmptaskans[2]
                    else:
                        est = max([local_aft_list[j] + computer_Cij(heft_local_answer_list, j, tmptask, tmptaskans[1])
                       for j in pre[tmptask]])

                    eft = est + workload[tmptask] / edge_computer_capability[tmptaskans[1]]

                    # match with local ava time list
                    match_task_time, match_intervel = match_time_interval(est, eft, local_ava_time, tmptaskans[1],
                                                     tmptask)

                    if match_task_time != None:
                        tmpanswer = [tmptask, match_task_time[0], match_task_time[1],
                                     match_task_time[2], match_intervel[0], match_intervel[1]]
                        local_running_task.append(tmptask)
                        local_running_answers.append(tmpanswer)

                        # reset the local ava time
                        reset_avatime_based_anslist(local_ava_time, tmpanswer, match_intervel[0],
                                                    match_intervel[1])

                        heft_local_answer_list[match_task_time[0]].append([tmptask, match_task_time[1],
                                                                           match_task_time[2]])

                        local_aft_list[tmptask] = match_task_time[2]
                else:
                    continue
            else:
                continue

        # step2: use the greedy to reschedule the local running tasks

        # reset all the ava time list
        local_ava_time = [[] for i in range(len(ava_time_list))]
        for i in range(len(ava_time_list)):
            for tmpavatime in ava_time_nlookback[i]:
                if tmpavatime[1] <= lookback_end_time and tmpavatime[0] >= lookback_start_time:
                    local_ava_time[i].append(tmpavatime.copy())
                else:
                    if tmpavatime[0] < lookback_end_time and tmpavatime[0] >= lookback_start_time:
                        if tmpavatime[0] - lookback_end_time != 0:
                            local_ava_time[i].append([tmpavatime[0], lookback_end_time])

                    if tmpavatime[0] < lookback_start_time:
                        if tmpavatime[1] > lookback_start_time and tmpavatime[1] <= lookback_end_time:
                            if tmpavatime[0] - lookback_end_time != 0:
                                local_ava_time[i].append([lookback_start_time, tmpavatime[1]])

        # reset all aft time list
        local_aft_list = [tmp for tmp in aft_list]

        # init the gredy answer list by copy the answer list
        greedy_answer_list = []
        greedy_answers = [[] for i  in range(len(ava_time_list))]
        for i in range(len(answer_list)):
            for tmp in answer_list[i]:
                greedy_answers[i].append(tmp.copy())

        for tmptask in local_running_task:
            first_task = tmptask
            eft_first_task_list = []
            # compute each processo using the insertion based time interval
            for i in range(len(ava_time_list)):
                local_ava_time[i].sort(key=lambda x: x[0], reverse=False)

                ready_time = 0
                if len(pre[first_task]) == 0:
                    ready_time = 0
                else:
                    ready_time = max(
                        [local_aft_list[j] + computer_Cij(greedy_answers, j, first_task, i) for j in pre[first_task]])
                early_ava_time = [-1, -1]
                for tmpavatime in local_ava_time[i]:
                    if tmpavatime[1] - ready_time >= workload[first_task] / edge_computer_capability[i]:
                        early_ava_time = [tmpavatime[0], tmpavatime[1]]
                        break
                # if len(pre[first_task]) == 0:
                #     est = 0
                # else:
                est = max([early_ava_time[0], ready_time])

                eft = workload[first_task] / edge_computer_capability[i] + est

                eft_first_task_list.append([i, est, eft, early_ava_time[0],
                                            early_ava_time[1]])

            # sor the list and choose the minimize eft of tasks
            eft_list = eft_first_task_list
            eft_list.sort(key=lambda x: x[2], reverse=False)

            # assign the task to the minimized eft task list
            minimize_processor = eft_list[0][0]
            task_best_time_intervel = [eft_list[0][1], eft_list[0][2]]

            # match the time intervel
            greedy_match_time, greedy_match_interval = match_time_interval(task_best_time_intervel[0],
                                                                           task_best_time_intervel[1],
                                                                           local_ava_time, minimize_processor,
                                                                           first_task)

            if greedy_match_time == None:
                greedy_answer_list = None
                break

            # reset the time intervel
            tmpanswer = [first_task, minimize_processor, greedy_match_time[1], greedy_match_time[2],
                         greedy_match_interval[0], greedy_match_interval[1]]
            reset_avatime_based_anslist(local_ava_time, tmpanswer,
                                        greedy_match_interval[0], greedy_match_interval[1])

            # add the answer to the greedy answer list
            greedy_answer_list.append(tmpanswer)
            greedy_answers[minimize_processor].append([first_task, tmpanswer[2], tmpanswer[3]])

            # reset the ava finsh time list
            local_aft_list[first_task] = greedy_match_time[2]

        # compute the greedy scheduling and choose the minized one
        if greedy_answer_list != None and len(greedy_answer_list) != 0:
            greedy_answer_list.sort(key=lambda x: x[3], reverse=True)
            local_running_answers.sort(key=lambda x: x[3], reverse=True)

            if local_running_answers[0][3] > greedy_answer_list[0][3]:
                # reset avatime answerlist aftlist based on heftonline
                reset_avatime_answerlist_aft(ava_time_nlookback, answer_list, aft_list, greedy_answer_list)
            else:
                # reset avatime answerlist aftlist based on greedy answer
                reset_avatime_answerlist_aft(ava_time_nlookback, answer_list, aft_list, local_running_answers)
        else:
            # reset avatime answerlist aft_list based on the heft oneline
            reset_avatime_answerlist_aft(ava_time_nlookback, answer_list, aft_list, local_running_answers)

        running_task += local_running_task
        lookback_start_time += n
        lookback_end_time += n
    return answer_list

def get_offline_answer(ans_list, task):
    for tmp in range(len(ans_list)):
        for tmpans in ans_list[tmp]:
            if tmpans[0] == task:
                return [task, tmp, tmpans[1], tmpans[2]]

    print("When find task {0} in anslist {1}, there can not be found".format(task, ans_list))
    return None

'heft offline 算法'
def heft_offline():
    import math
    import numpy as np
    from queue import Queue
    import sys

    # set computation cost and communication cost
    computation_cost = [np.average([tmpw / tmpC for tmpC in edge_computer_capability]) for tmpw in workload]

    communication_cost = np.array(datasize).copy()
    W_list = []
    for tmp in W:
        W_list += tmp
    for i in range(len(datasize)):
        for j in range(len(datasize[i])):
            communication_cost[i][j] = np.average([communication_cost[i][j] / tmp for tmp in W_list])

    # computer the rank upward for all traversing task graph upward starting from the exit task
    task_uprank = [0 for i in range(len(workload))]

    # last_task = -1  # find the last task
    # for i, tmp in enumerate(succ):
    #     if len(tmp) == 0:
    #         last_task = i
    # tranverse_queue = Queue()
    # visited = set()

    post_order_sequence = first_order_sequence(pre,succ)
    post_order_sequence.reverse()
    for tmp in post_order_sequence:
        if len(succ[tmp]) == 0:
            task_uprank[tmp] = computation_cost[tmp]
        else:
            task_uprank[tmp] = computation_cost[tmp] + np.max(
                [communication_cost[tmp][order2subtaskindex_map[j]] + task_uprank[j] for j in succ[tmp]])

    task_uprank = [[i, tmp] for i, tmp in enumerate(task_uprank)]
    task_schedule_sequence = [i for i, tmp in enumerate(task_uprank)]

    # sort the tasks scheduling list by no increaseing
    task_uprank.sort(reverse=True, key=takeSecond)

    # init the answer list
    answer_list = [[] for i in range(len(ava_time_list))]
    ava_time_now = [[[0, sys.maxsize]] for i in range(len(ava_time_list))]  # copy the ava_time_list
    # for i in range(len(ava_time_list)):
    #     for tmp in ava_time_list[i]:
    #         ava_time_now[i].append(tmp.copy())
    #     ava_time_now[i].sort(key=takeSecond, reverse=False)

    # AFT list the actual finish time list
    aft_list = [-1 for i in range(len(workload))]

    while len(task_uprank) != 0:
        first_task = task_uprank[0][0]  # select the first task

        eft_first_task_list = []
        # compute the eft for each processor using the insertion-based scheduling policy
        for i in range(len(ava_time_list)):
            ava_time_now[i].sort(key=lambda x: x[0], reverse=False)
            if len(pre[first_task]) == 0:
                est = 0
            else:
                est = max([ava_time_now[i][0][0],
                           max([aft_list[j] + computer_Cij(answer_list, j, first_task, i) for j in pre[first_task]])])

            eft = workload[first_task] / edge_computer_capability[i] + est

            eft_first_task_list.append([i, est, eft])

        # sor the list and choose the minimize eft of tasks
        eft_list = eft_first_task_list
        eft_list.sort(key=lambda x: x[2], reverse=False)

        # assign the task to the minimized eft task list
        minimize_processor = eft_list[0][0]
        task_best_time_intervel = [eft_list[0][1], eft_list[0][2]]
        match = False
        for i, tmp in enumerate(ava_time_now[minimize_processor]):
            start_time = -1
            end_time = -1

            # the best time intervel match case
            if tmp[0] <= task_best_time_intervel[0] and tmp[1] >= task_best_time_intervel[1]:  # case 1
                match = True
                start_time = task_best_time_intervel[0]
                end_time = task_best_time_intervel[1]

            if tmp[0] <= task_best_time_intervel[0] and tmp[1] < task_best_time_intervel[1]:  # case 2
                pass
            if tmp[1] >= task_best_time_intervel[0]:  # case 3
                pass

            if tmp[0] >= task_best_time_intervel[0]:  # case 4 and case 5
                if workload[first_task] / edge_computer_capability[minimize_processor] <= tmp[1] - tmp[0]:
                    match = True
                    start_time = tmp[0]
                    end_time = tmp[0] + workload[first_task] / edge_computer_capability[minimize_processor]
                else:
                    pass

            if match:
                # change the avatime
                if tmp[0] == start_time:
                    ava_time_now[minimize_processor][i][0] = end_time
                elif tmp[1] == end_time:
                    ava_time_now[minimize_processor][i][1] = start_time

                if tmp[0] < start_time and tmp[1] > end_time:
                    tmpendtime = tmp[1]
                    ava_time_now[minimize_processor][i][0] = end_time
                    # ava_time_now[minimize_processor].append([end_time, tmpendtime])

                # add to the answer
                answer_list[minimize_processor].append([first_task, start_time, end_time])

                # change the finish time list
                aft_list[first_task] = end_time

                break

        if not match:
            print("Can not find a ava for task {0} from start time {1} to end time {2}".format(first_task,
                                                                                               task_best_time_intervel[
                                                                                                   0],
                                                                                               task_best_time_intervel[
                                                                                                   1]))
            return None
        task_uprank.remove(task_uprank[0])

    return answer_list

'nsga offline algorithm'
def NSGA_offline():
    # main
    import sys
    population = 10
    visit_order = first_order_sequence(pre,succ)
    individual_list = []
    top_ratio = 0.1
    crosscover_ratio = 0.65
    mutation_ratio = 0.1
    iter_size = 2
    iter_number = 0
    best_individual = None 
    best_ans_dict = None
    best_fitness = float("inf")
    for i in range(population):
        individual_list.append(chromosome_code(visit_order))

    while iter_number < iter_size:
        parents_list,local_fitness,local_ans_dict,local_answear = selection(individual_list)
        offsprings_list = generate_offspring(individual_list,parents_list,population,crosscover_ratio,mutation_ratio)
        
        if best_fitness > local_fitness:
            best_individual_alternative = individual_list[parents_list[0]]
            best_ans_dict = local_ans_dict
            best_fitness = local_fitness
            nsga_answear = local_answear
        iter_number += 1
        individual_list = offsprings_list

    return nsga_answear


def reset_avatime_answerlist_aft(avatimelist, answerlist, aftlist, anslist):
    # reset answerlist
    for tmp in anslist:
        answerlist[tmp[1]].append([tmp[0], tmp[2], tmp[3]])

    # reset aftlist
    for tmp in anslist:
        aftlist[tmp[0]] = tmp[3]

    # reset ava time list
    for tmp in anslist:
        reset_avatime_based_anslist(avatimelist, tmp, tmp[4], tmp[5])

def match_time_interval(start, end, time_interval_list, device, task):

    task_best_time_intervel = [start, end]
    first_task = task
    minimize_processor = device
    match = False
    if device != None: # match the special device of time interval
        time_interval_list[device].sort(key=lambda x: x[0], reverse=False)
        for i,tmp in enumerate(time_interval_list[device]):
            start_time = -1
            end_time = -1

            # if match the best time case
            # the best time intervel match case
            if tmp[0] <= task_best_time_intervel[0] and tmp[1] >= task_best_time_intervel[1]:  # case 1
                match = True
                start_time = task_best_time_intervel[0]
                end_time = task_best_time_intervel[1]

            if tmp[0] <= task_best_time_intervel[0] and tmp[1] < task_best_time_intervel[1]:  # case 2
                pass
            if tmp[1] >= task_best_time_intervel[0]:  # case 3
                pass

            if tmp[0] >= task_best_time_intervel[0]:  # case 4 and case 5
                if workload[first_task] / edge_computer_capability[minimize_processor] <= tmp[1] - tmp[0]:
                    match = True
                    start_time = tmp[0]
                    end_time = tmp[0] + workload[first_task] / edge_computer_capability[minimize_processor]
                else:
                    pass

            if match:
                return [device, start_time, end_time], [tmp[0], tmp[1]]

        if not match:
            # print("Can not find a ava for task {0} from start time {1} to end time {2}".format(first_task,
            #                                                                                    task_best_time_intervel[
            #                                                                                        0],
            #                                                                                    task_best_time_intervel[
            #                                                                                       1]))
            return None, None
    else: # match the all device of time interval
        for j in range(len(time_interval_list)):
            minimize_processor = j
            time_interval_list[minimize_processor].sort(key=lambda x:x[0], reverse=False)
            for i, tmp in enumerate(time_interval_list[minimize_processor]):
                start_time = -1
                end_time = -1

                # if match the best time case
                # the best time intervel match case
                if tmp[0] <= task_best_time_intervel[0] and tmp[1] >= task_best_time_intervel[1]:  # case 1
                    match = True
                    start_time = task_best_time_intervel[0]
                    end_time = task_best_time_intervel[1]

                if tmp[0] <= task_best_time_intervel[0] and tmp[1] < task_best_time_intervel[1]:  # case 2
                    pass
                if tmp[1] >= task_best_time_intervel[0]:  # case 3
                    pass

                if tmp[0] >= task_best_time_intervel[0]:  # case 4 and case 5
                    if workload[first_task] / edge_computer_capability[minimize_processor] <= tmp[1] - tmp[0]:
                        match = True
                        start_time = tmp[0]
                        end_time = tmp[0] + workload[first_task] / edge_computer_capability[minimize_processor]
                    else:
                        pass

                if match:
                    return [device, start_time, end_time], tmp

        # print("Can not find a ava for task {0} from start time {1} to end time {2}".format(first_task,
        #                                                                                    task_best_time_intervel[
        #                                                                                        0],
        #                                                                                    task_best_time_intervel[
        #                                                                                        1]))
        return None, None

def reset_avatime_based_anslist(avatimelist, ans, interval_start, interval_end):

    start_time = ans[2]
    end_time = ans[3]
    minimize_processor = ans[1]

    for i,tmp in enumerate(avatimelist[ans[1]]):

        if  interval_start < tmp[1]:
            # change the avatime
            if tmp[0] == start_time:
                avatimelist[minimize_processor][i][0] = end_time
            elif tmp[1] == end_time:
                avatimelist[minimize_processor][i][1] = start_time

            if tmp[0] < start_time and tmp[1] > end_time:
                tmpendtime = tmp[1]
                avatimelist[minimize_processor][i][1] = start_time
                avatimelist[minimize_processor].append([end_time, tmpendtime])
                avatimelist[minimize_processor].sort(key=lambda x: x[0], reverse=False)
                # print()
            break
    # return avatimelist

def reservation_reallocate_avatime(ava_time_list,unava_time_list):
    ava_time_intervals = [[] for i in range(len(ava_time_list))]  # copy the ava_time_list

    for i in range(len(ava_time_list)):
        for epoch_k in range(round(len(ava_time_list[i])/2)):
            unava_len = unava_time_list[i][epoch_k][1] - unava_time_list[i][epoch_k][0] 
            tmpava_end = ava_time_list[i][epoch_k*2+1][1] - unava_len
            tmpava_start = ava_time_list[i][epoch_k*2][0]
            tmp_interval = [tmpava_start,tmpava_end]
            ava_time_intervals[i].append(tmp_interval)

    return ava_time_intervals

# online imporved
def greedy_time_reservation_nlook_back_improved(n):

    # copy teh ava time list
    # ava_time_nlookback = [[] for i in range(len(ava_time_list))]
    # for i in range(len(ava_time_list)):
    #     for j in range(len(ava_time_list[i])):
    #         ava_time_nlookback[i].append(ava_time_list[i][j].copy())

    ava_time_nlookback = reservation_reallocate_avatime(ava_time_list,decision_time_list)
    # print(ava_time_nlookback)
    # init parameters
    running_task = []  # has running task in the look back time
    visited_order = first_order_sequence(pre,succ)  # the first order visited orderf
    aft_list = [-1 for i in range(len(workload))]  # the actual finish time for each task
    answer_list = [[] for i in range(len(ava_time_list))]  # the total answer list

    # for 0 ~ n, get
    lookback_start_time = 0
    lookback_end_time = n
    while len(running_task) != len(workload):

        # reset time interval
        local_ava_time = [[] for i in range(len(ava_time_list))]
        for i in range(len(ava_time_list)):
            for tmpavatime in ava_time_nlookback[i]:
                if tmpavatime[1] <= lookback_end_time and tmpavatime[0] >= lookback_start_time:
                    local_ava_time[i].append(tmpavatime.copy())
                else:
                    if tmpavatime[0] < lookback_end_time and tmpavatime[0] >= lookback_start_time:
                        if tmpavatime[0] - lookback_end_time != 0:
                            local_ava_time[i].append([tmpavatime[0], lookback_end_time])

                    if tmpavatime[0] < lookback_start_time:
                        if tmpavatime[1] > lookback_start_time and tmpavatime[1] <= lookback_end_time:
                            if tmpavatime[0] - lookback_end_time != 0:
                                local_ava_time[i].append([lookback_start_time, tmpavatime[1]])
                    if tmpavatime[0] < lookback_start_time and tmpavatime[1] > lookback_end_time:
                        local_ava_time[i].append([lookback_start_time, lookback_end_time])
        # local running task
        local_running_task = []
        # local running answer
        local_running_answers = []
        # local task finish time
        local_aft_list = [tmp for tmp in aft_list]

        # greedy anser list init with copy with answerlt
        greedy_local_answer_list = [[] for i in range(len(ava_time_list))]
        for i in range(len(answer_list)):
            for tmptimeinterval in answer_list[i]:
                greedy_local_answer_list[i].append(tmptimeinterval.copy())

        # choose the task from the unfinish task and use greedy to dispatch the tasks
        for tmptask in visited_order:
            if tmptask not in running_task and tmptask not in local_running_task:
                # discrimite pre task is done
                pre_done = True
                for tmppretask in pre[tmptask]:
                    if tmppretask not in running_task and tmppretask not in local_running_task:
                        pre_done = False

                if pre_done:
                    # use the greedy to dispatch the tasks
                    first_task = tmptask

                    eft_first_task_list = []
                    # compute the eft for each processor using the insertion-based scheduling policy
                    for i in range(len(ava_time_list)):

                        if len(local_ava_time[i]) == 0:
                            continue
                        local_ava_time[i].sort(key=lambda x: x[0], reverse=False)

                        ready_time = 0
                        if len(pre[first_task]) == 0 or pre[first_task][0] == 0:
                            ready_time = 0
                        else:
                            ready_time = max(
                                [local_aft_list[j] + computer_Cij(greedy_local_answer_list, j, first_task, i) for j in pre[first_task]])
                        early_ava_time = -1
                        for tmpavatime in local_ava_time[i]:
                            if tmpavatime[1] - ready_time >= workload[first_task] / edge_computer_capability[i]:
                                if tmpavatime[0] + workload[first_task] / edge_computer_capability[i] <= tmpavatime[1]:
                                    early_ava_time = tmpavatime[0]
                                    break
                        if early_ava_time != -1:
                            est = max([early_ava_time, ready_time])

                            eft = workload[first_task] / edge_computer_capability[i] + est

                            eft_first_task_list.append([i, est, eft])

                    # sor the list and choose the minimize eft of tasks
                    eft_list = eft_first_task_list
                    eft_list.sort(key=lambda x: x[2], reverse=False)

                    if len(eft_list) == 0:
                        continue

                    # assign the task to the minimized eft task list
                    minimize_processor = eft_list[0][0]
                    task_best_time_intervel = [eft_list[0][1], eft_list[0][2]]

                    match_time, match_interval = match_time_interval(task_best_time_intervel[0],
                                                                     task_best_time_intervel[1],
                                                                     local_ava_time, minimize_processor,
                                                                     first_task)

                    if match_time != None:
                        # reset all the things
                        tmpanswer = [tmptask, match_time[0], match_time[1],
                                     match_time[2], match_interval[0], match_interval]

                        local_running_task.append(tmptask)
                        local_running_answers.append(tmpanswer)

                        # reset the local ava time
                        reset_avatime_based_anslist(local_ava_time, tmpanswer, match_interval[0],
                                                    match_interval[1])

                        greedy_local_answer_list[match_time[0]].append([tmptask, match_time[1],
                                                                           match_time[2]])

                        local_aft_list[tmptask] = match_time[2]
                    else:
                        continue

                else:
                    continue

        # reset the n look back time
        reset_avatime_answerlist_aft(ava_time_nlookback, answer_list, aft_list, local_running_answers)

        running_task += local_running_task
        lookback_start_time += n
        lookback_end_time += n

    return answer_list

def greedy_nlook_back_improved(n):

    # copy teh ava time list
    ava_time_nlookback = [[] for i in range(len(ava_time_list))]
    for i in range(len(ava_time_list)):
        for j in range(len(ava_time_list[i])):
            ava_time_nlookback[i].append(ava_time_list[i][j].copy())

    # init parameters
    running_task = []  # has running task in the look back time
    visited_order = first_order_sequence(pre,succ)  # the first order visited orderf
    aft_list = [-1 for i in range(len(workload))]  # the actual finish time for each task
    answer_list = [[] for i in range(len(ava_time_list))]  # the total answer list

    # for 0 ~ n, get
    lookback_start_time = 0
    lookback_end_time = n
    while len(running_task) != len(workload):

        # reset time interval
        local_ava_time = [[] for i in range(len(ava_time_list))]
        for i in range(len(ava_time_list)):
            for tmpavatime in ava_time_nlookback[i]:
                if tmpavatime[1] <= lookback_end_time and tmpavatime[0] >= lookback_start_time:
                    local_ava_time[i].append(tmpavatime.copy())
                else:
                    if tmpavatime[0] < lookback_end_time and tmpavatime[0] >= lookback_start_time:
                        if tmpavatime[0] - lookback_end_time != 0:
                            local_ava_time[i].append([tmpavatime[0], lookback_end_time])

                    if tmpavatime[0] < lookback_start_time:
                        if tmpavatime[1] > lookback_start_time and tmpavatime[1] <= lookback_end_time:
                            if tmpavatime[0] - lookback_end_time != 0:
                                local_ava_time[i].append([lookback_start_time, tmpavatime[1]])
                    if tmpavatime[0] < lookback_start_time and tmpavatime[1] > lookback_end_time:
                        local_ava_time[i].append([lookback_start_time, lookback_end_time])
        # local running task
        local_running_task = []
        # local running answer
        local_running_answers = []
        # local task finish time
        local_aft_list = [tmp for tmp in aft_list]

        # greedy anser list init with copy with answerlt
        greedy_local_answer_list = [[] for i in range(len(ava_time_list))]
        for i in range(len(answer_list)):
            for tmptimeinterval in answer_list[i]:
                greedy_local_answer_list[i].append(tmptimeinterval.copy())

        # choose the task from the unfinish task and use greedy to dispatch the tasks
        for tmptask in visited_order:
            if tmptask not in running_task and tmptask not in local_running_task:
                # discrimite pre task is done
                pre_done = True
                for tmppretask in pre[tmptask]:
                    if tmppretask not in running_task and tmppretask not in local_running_task:
                        pre_done = False

                if pre_done:
                    # use the greedy to dispatch the tasks
                    first_task = tmptask

                    eft_first_task_list = []
                    # compute the eft for each processor using the insertion-based scheduling policy
                    for i in range(len(ava_time_list)):

                        if len(local_ava_time[i]) == 0:
                            continue
                        local_ava_time[i].sort(key=lambda x: x[0], reverse=False)

                        ready_time = 0
                        if len(pre[first_task]) == 0 or pre[first_task][0] == 0:
                            ready_time = 0
                        else:
                            ready_time = max(
                                [local_aft_list[j] + computer_Cij(greedy_local_answer_list, j, first_task, i) for j in pre[first_task]])
                        early_ava_time = -1
                        for tmpavatime in local_ava_time[i]:
                            if tmpavatime[1] - ready_time >= workload[first_task] / edge_computer_capability[i]:
                                if tmpavatime[0] + workload[first_task] / edge_computer_capability[i] <= tmpavatime[1]:
                                    early_ava_time = tmpavatime[0]
                                    break
                        if early_ava_time != -1:
                            est = max([early_ava_time, ready_time])

                            eft = workload[first_task] / edge_computer_capability[i] + est

                            eft_first_task_list.append([i, est, eft])

                    # sor the list and choose the minimize eft of tasks
                    eft_list = eft_first_task_list
                    eft_list.sort(key=lambda x: x[2], reverse=False)

                    if len(eft_list) == 0:
                        continue

                    # assign the task to the minimized eft task list
                    minimize_processor = eft_list[0][0]
                    task_best_time_intervel = [eft_list[0][1], eft_list[0][2]]

                    match_time, match_interval = match_time_interval(task_best_time_intervel[0],
                                                                     task_best_time_intervel[1],
                                                                     local_ava_time, minimize_processor,
                                                                     first_task)

                    if match_time != None:
                        # reset all the things
                        tmpanswer = [tmptask, match_time[0], match_time[1],
                                     match_time[2], match_interval[0], match_interval[1]]

                        local_running_task.append(tmptask)
                        local_running_answers.append(tmpanswer)

                        # reset the local ava time
                        reset_avatime_based_anslist(local_ava_time, tmpanswer, match_interval[0],
                                                    match_interval[1])

                        greedy_local_answer_list[match_time[0]].append([tmptask, match_time[1],
                                                                           match_time[2]])

                        local_aft_list[tmptask] = match_time[2]
                    else:
                        continue

                else:
                    continue

        # reset the n look back time
        reset_avatime_answerlist_aft(ava_time_nlookback, answer_list, aft_list, local_running_answers)

        running_task += local_running_task
        lookback_start_time += n
        lookback_end_time += n

    return answer_list

def heft_n_look_back_improved(n):
    # based on heft, get a anser list
    heft_ans = heft_offline()

    # copy the ava_time_list
    ava_time_nlookback = [[] for i in range(len(ava_time_list))]
    for i in range(len(ava_time_list)):
        for j in range(len(ava_time_list[i])):
            ava_time_nlookback[i].append(ava_time_list[i][j].copy())

    # init parameters
    running_task = []  # has running task in the look back time
    visited_order = first_order_sequence(pre,succ)  # the first order visited orderf
    aft_list = [-1 for i in range(len(workload))]  # the actual finish time for each task
    answer_list = [[] for i in range(len(ava_time_list))]  # the total answer list

    # for 0 ~ n, get
    lookback_start_time = 0
    lookback_end_time = n
    while len(running_task) != len(workload):
        # reset time interval
        local_ava_time = [[] for i in range(len(ava_time_list))]
        for i in range(len(ava_time_list)):
            for tmpavatime in ava_time_nlookback[i]:
                if tmpavatime[1] <= lookback_end_time and tmpavatime[0] >= lookback_start_time:
                    local_ava_time[i].append(tmpavatime.copy())
                else:
                    if tmpavatime[0] < lookback_end_time and tmpavatime[0] >= lookback_start_time:
                        if tmpavatime[0] - lookback_end_time != 0:
                            local_ava_time[i].append([tmpavatime[0], lookback_end_time])

                    if tmpavatime[0] < lookback_start_time:
                        if tmpavatime[1] > lookback_start_time and tmpavatime[1] <= lookback_end_time:
                            if tmpavatime[0] - lookback_end_time != 0:
                                local_ava_time[i].append([lookback_start_time, tmpavatime[1]])

        # local running task
        local_running_task = []
        # local running answer
        local_running_answers = []
        # local task finish time
        local_aft_list = [tmp for tmp in aft_list]

        # heft answer list init with copy with answer list
        heft_local_answer_list = [[] for i in range(len(ava_time_list))]
        for i in range(len(answer_list)):
            for tmptimeinterval in answer_list[i]:
                heft_local_answer_list[i].append(tmptimeinterval.copy())

        # choose the tasks from unfinish task and get heft offline answer, task finish time
        for tmptask in visited_order:
            if tmptask not in running_task and tmptask not in local_running_task:  # chooice the task from the unfinish task
                # get offline answer
                tmptaskans = get_offline_answer(heft_ans, tmptask)

                # if all the pre task has been done
                pre_done = True
                for tmppretask in pre[tmptask]:
                    if tmppretask not in running_task and tmppretask not in local_running_task:
                        pre_done = False

                if pre_done:
                    ava_timw_now = local_ava_time[tmptaskans[1]]

                    if len(ava_timw_now) == 0:
                        break

                    if len(pre[tmptask]) == 0:
                        est = tmptaskans[2]
                    elif pre[tmptask][0]==0:
                        est = ava_timw_now[0][0]
                    else:
                        est = max([local_aft_list[j] + computer_Cij(heft_local_answer_list, j, tmptask, tmptaskans[1])
                                   for j in pre[tmptask]])

                    eft = est + workload[tmptask] / edge_computer_capability[tmptaskans[1]]

                    # match with local ava time list
                    match_task_time, match_intervel = match_time_interval(est, eft, local_ava_time, tmptaskans[1],
                                                                          tmptask)

                    if match_task_time != None:
                        tmpanswer = [tmptask, match_task_time[0], match_task_time[1],
                                     match_task_time[2], match_intervel[0], match_intervel[1]]
                        local_running_task.append(tmptask)
                        local_running_answers.append(tmpanswer)

                        # reset the local ava time
                        reset_avatime_based_anslist(local_ava_time, tmpanswer, match_intervel[0],
                                                    match_intervel[1])

                        heft_local_answer_list[match_task_time[0]].append([tmptask, match_task_time[1],
                                                                           match_task_time[2]])

                        local_aft_list[tmptask] = match_task_time[2]
                else:
                    continue
            else:
                continue

        reset_avatime_answerlist_aft(ava_time_nlookback, answer_list, aft_list, local_running_answers)

        running_task += local_running_task
        lookback_start_time += n
        lookback_end_time += n
    return answer_list

def NSGA_n_look_back_improved(n):
    pass

# online
def greedy_nlook_back(n):
    ava_time_nlookback = [[] for i in range(len(ava_time_list))]
    for i in range(len(ava_time_list)):
        for j in range(len(ava_time_list[i])):
            ava_time_nlookback[i].append(ava_time_list[i][j].copy())
    # print(ava_time_nlookback)
    # init parameters
    running_task = []  # has running task in the look back time
    visited_order = first_order_sequence(pre,succ)  # the first order visited orderf
    aft_list = [-1 for i in range(len(workload))]  # the actual finish time for each task
    answer_list = [[] for i in range(len(ava_time_list))]  # the total answer list

    # for 0 ~ n, get
    lookback_start_time = 0
    lookback_end_time = n
    tmpsubtask_index = 0

    while tmpsubtask_index != len(workload)-1:
        # reset local time interval
        local_ava_time = [[] for i in range(len(ava_time_list))]
        for i in range(len(ava_time_list)):
            for tmpavatime in ava_time_nlookback[i]:
                if tmpavatime[1] <= lookback_end_time and tmpavatime[0] >= lookback_start_time:
                    local_ava_time[i].append(tmpavatime.copy())
                else:
                    if tmpavatime[0] < lookback_end_time and tmpavatime[0] >= lookback_start_time:
                        if tmpavatime[0] - lookback_end_time != 0:
                            local_ava_time[i].append([tmpavatime[0], lookback_end_time])

                    if tmpavatime[0] < lookback_start_time:
                        if tmpavatime[1] > lookback_start_time and tmpavatime[1] <= lookback_end_time:
                            if tmpavatime[0] - lookback_end_time != 0:
                                local_ava_time[i].append([lookback_start_time, tmpavatime[1]])
                    if tmpavatime[0] < lookback_start_time and tmpavatime[1] > lookback_end_time:
                        local_ava_time[i].append([lookback_start_time, lookback_end_time])
        
        # local running task
        # local_running_task = []
        # local running answer
        local_running_answers = []
        # local task finish time
        local_aft_list = [tmp for tmp in aft_list]

        # greedy anser list init with copy with answerlt
        greedy_local_answer_list = [[] for i in range(len(ava_time_list))]
        for i in range(len(answer_list)):
            for tmptimeinterval in answer_list[i]:
                greedy_local_answer_list[i].append(tmptimeinterval.copy())

        # choose the task from the unfinish task and use greedy to dispatch the tasks
        while tmpsubtask_index != len(workload)-1:
            # use the greedy to dispatch the tasks

            first_task = visited_order[tmpsubtask_index]
            # if first_task == 37:
            #     print()
            eft_first_task_list = []
            # compute the eft for each processor using the insertion-based scheduling policy
            for i in range(len(ava_time_list)):
                if len(local_ava_time[i]) == 0:
                    continue
                local_ava_time[i].sort(key=lambda x: x[0], reverse=False)

                ready_time = 0
                if len(pre[first_task]) == 0 or pre[first_task][0] == 0:
                    ready_time = 0
                else:
                    ready_time = max(
                        [local_aft_list[j] + computer_Cij(greedy_local_answer_list, j, first_task, i) for j in pre[first_task]])
                early_ava_time = -1

                for tmpavatime in local_ava_time[i]:
                    if tmpavatime[1] - ready_time >= workload[first_task] / edge_computer_capability[i]:
                        if tmpavatime[0] + workload[first_task] / edge_computer_capability[i] <= tmpavatime[1]:
                            early_ava_time = tmpavatime[0]
                            break
                if early_ava_time != -1:
                    est = max([early_ava_time, ready_time])

                    eft = workload[first_task] / edge_computer_capability[i] + est

                    eft_first_task_list.append([i, est, eft])

            # sor the list and choose the minimize eft of tasks
            eft_list = eft_first_task_list
            eft_list.sort(key=lambda x: x[2], reverse=False)

            if len(eft_list) == 0:
                break

            # assign the task to the minimized eft task list
            minimize_processor = eft_list[0][0]
            task_best_time_intervel = [eft_list[0][1], eft_list[0][2]]

            match_time, match_interval = match_time_interval(task_best_time_intervel[0],
                                                                task_best_time_intervel[1],
                                                                local_ava_time, minimize_processor,
                                                                first_task)

            if match_time != None:
                # reset all the things
                tmpanswer = [first_task, match_time[0], match_time[1],
                                match_time[2], match_interval[0], match_interval[1]]

                # local_running_task.append(first_task)
                local_running_answers.append(tmpanswer)

                # reset the local ava time
                reset_avatime_based_anslist(local_ava_time, tmpanswer, match_interval[0],
                                            match_interval[1])

                greedy_local_answer_list[match_time[0]].append([first_task, match_time[1],
                                                                    match_time[2]])

                local_aft_list[first_task] = match_time[2]

                tmpsubtask_index += 1
            else:
                break

        # reset the n look back time
        reset_avatime_answerlist_aft(ava_time_nlookback, answer_list, aft_list, local_running_answers)

        # running_task += local_running_task
        lookback_start_time += n
        lookback_end_time += n


    return answer_list

def greedy_time_reservation_nlook_back(n):

    ava_time_nlookback = reservation_reallocate_avatime(ava_time_list,decision_time_list)
    # print(ava_time_nlookback)
    # init parameters
    running_task = []  # has running task in the look back time
    visited_order = first_order_sequence(pre,succ)  # the first order visited orderf
    aft_list = [-1 for i in range(len(workload))]  # the actual finish time for each task
    answer_list = [[] for i in range(len(ava_time_list))]  # the total answer list

    # for 0 ~ n, get
    lookback_start_time = 0
    lookback_end_time = n
    tmpsubtask_index = 0

    while tmpsubtask_index != len(workload)-1:
        # reset local time interval
        local_ava_time = [[] for i in range(len(ava_time_list))]
        for i in range(len(ava_time_list)):
            for tmpavatime in ava_time_nlookback[i]:
                if tmpavatime[1] <= lookback_end_time and tmpavatime[0] >= lookback_start_time:
                    local_ava_time[i].append(tmpavatime.copy())
                else:
                    if tmpavatime[0] < lookback_end_time and tmpavatime[0] >= lookback_start_time:
                        if tmpavatime[0] - lookback_end_time != 0:
                            local_ava_time[i].append([tmpavatime[0], lookback_end_time])

                    if tmpavatime[0] < lookback_start_time:
                        if tmpavatime[1] > lookback_start_time and tmpavatime[1] <= lookback_end_time:
                            if tmpavatime[0] - lookback_end_time != 0:
                                local_ava_time[i].append([lookback_start_time, tmpavatime[1]])
                    if tmpavatime[0] < lookback_start_time and tmpavatime[1] > lookback_end_time:
                        local_ava_time[i].append([lookback_start_time, lookback_end_time])
        
        # local running task
        # local_running_task = []
        # local running answer
        local_running_answers = []
        # local task finish time
        local_aft_list = [tmp for tmp in aft_list]

        # greedy anser list init with copy with answerlt
        greedy_local_answer_list = [[] for i in range(len(ava_time_list))]
        for i in range(len(answer_list)):
            for tmptimeinterval in answer_list[i]:
                greedy_local_answer_list[i].append(tmptimeinterval.copy())

        # choose the task from the unfinish task and use greedy to dispatch the tasks
        while tmpsubtask_index != len(workload)-1:
            # use the greedy to dispatch the tasks
            first_task = visited_order[tmpsubtask_index]
            
            eft_first_task_list = []
            # compute the eft for each processor using the insertion-based scheduling policy
            for i in range(len(ava_time_list)):
                if len(local_ava_time[i]) == 0:
                    continue
                local_ava_time[i].sort(key=lambda x: x[0], reverse=False)

                ready_time = 0
                if len(pre[first_task]) == 0 or pre[first_task][0] == 0:
                    ready_time = 0
                else:
                    ready_time = max(
                        [local_aft_list[j] + computer_Cij(greedy_local_answer_list, j, first_task, i) for j in pre[first_task]])
                early_ava_time = -1

                for tmpavatime in local_ava_time[i]:
                    if tmpavatime[1] - ready_time >= workload[first_task] / edge_computer_capability[i]:
                        if tmpavatime[0] + workload[first_task] / edge_computer_capability[i] <= tmpavatime[1]:
                            early_ava_time = tmpavatime[0]
                            break

                if early_ava_time != -1:
                    est = max([early_ava_time, ready_time])

                    eft = workload[first_task] / edge_computer_capability[i] + est

                    eft_first_task_list.append([i, est, eft])

            # sor the list and choose the minimize eft of tasks
            eft_list = eft_first_task_list
            eft_list.sort(key=lambda x: x[2], reverse=False)

            if len(eft_list) == 0:
                break

            # assign the task to the minimized eft task list
            minimize_processor = eft_list[0][0]
            task_best_time_intervel = [eft_list[0][1], eft_list[0][2]]

            match_time, match_interval = match_time_interval(task_best_time_intervel[0],
                                                                task_best_time_intervel[1],
                                                                local_ava_time, minimize_processor,
                                                                first_task)

            if match_time != None:
                # reset all the things
                tmpanswer = [first_task, match_time[0], match_time[1],
                                match_time[2], match_interval[0], match_interval[1]]

                # local_running_task.append(first_task)
                local_running_answers.append(tmpanswer)

                # reset the local ava time
                reset_avatime_based_anslist(local_ava_time, tmpanswer, match_interval[0],
                                            match_interval[1])

                greedy_local_answer_list[match_time[0]].append([first_task, match_time[1],
                                                                    match_time[2]])

                local_aft_list[first_task] = match_time[2]

                tmpsubtask_index += 1
            else:
                break

        # reset the n look back time
        reset_avatime_answerlist_aft(ava_time_nlookback, answer_list, aft_list, local_running_answers)

        # running_task += local_running_task
        lookback_start_time += n
        lookback_end_time += n


    return answer_list

def heft_n_look_back(n):
    # based on heft, get a anser list
    heft_ans = heft_offline()

    # copy the ava_time_list
    ava_time_nlookback = [[] for i in range(len(ava_time_list))]
    for i in range(len(ava_time_list)):
        for j in range(len(ava_time_list[i])):
            ava_time_nlookback[i].append(ava_time_list[i][j].copy())

    # init parameters
    # running_task = []  # has running task in the look back time
    visited_order = first_order_sequence(pre,succ)  # the first order visited orderf
    aft_list = [-1 for i in range(len(workload))]  # the actual finish time for each task
    answer_list = [[] for i in range(len(ava_time_list))]  # the total answer list

    # for 0 ~ n, get
    lookback_start_time = 0
    lookback_end_time = n
    tmpsubtask_index = 0

    while tmpsubtask_index != len(workload)-1:
        # reset time interval
        local_ava_time = [[] for i in range(len(ava_time_list))]
        for i in range(len(ava_time_list)):
            for tmpavatime in ava_time_nlookback[i]:
                if tmpavatime[1] <= lookback_end_time and tmpavatime[0] >= lookback_start_time:
                    local_ava_time[i].append(tmpavatime.copy())
                else:
                    if tmpavatime[0] < lookback_end_time and tmpavatime[0] >= lookback_start_time:
                        if tmpavatime[0] - lookback_end_time != 0:
                            local_ava_time[i].append([tmpavatime[0], lookback_end_time])

                    if tmpavatime[0] < lookback_start_time:
                        if tmpavatime[1] > lookback_start_time and tmpavatime[1] <= lookback_end_time:
                            if tmpavatime[0] - lookback_end_time != 0:
                                local_ava_time[i].append([lookback_start_time, tmpavatime[1]])
                    if tmpavatime[0] < lookback_start_time and tmpavatime[1] > lookback_end_time:
                        local_ava_time[i].append([lookback_start_time, lookback_end_time])
        # local running task
        # local_running_task = []
        # local running answer
        local_running_answers = []
        # local task finish time
        local_aft_list = [tmp for tmp in aft_list]

        # heft answer list init with copy with answer list
        heft_local_answer_list = [[] for i in range(len(ava_time_list))]
        for i in range(len(answer_list)):
            for tmptimeinterval in answer_list[i]:
                heft_local_answer_list[i].append(tmptimeinterval.copy())


        # choose the tasks from unfinish task and get heft offline answer, task finish time
        while tmpsubtask_index != len(workload)-1:

            tmptask = visited_order[tmpsubtask_index]

            tmptaskans = get_offline_answer(heft_ans, tmptask)

            ava_timw_now = local_ava_time[tmptaskans[1]]

            if len(ava_timw_now) == 0:
                break

            if len(pre[tmptask]) == 0:
                est = 0
            elif pre[tmptask][0] == 0:
                est = ava_timw_now[0][0]
            else:
                est = max([local_aft_list[j] + computer_Cij(heft_local_answer_list, j, tmptask, tmptaskans[1])
                            for j in pre[tmptask]])

            eft = est + workload[tmptask] / edge_computer_capability[tmptaskans[1]]

            # match with local ava time list
            match_task_time, match_intervel = match_time_interval(est, eft, local_ava_time, tmptaskans[1],
                                                                    tmptask)

            if match_task_time != None:
                tmpanswer = [tmptask, match_task_time[0], match_task_time[1],
                                match_task_time[2], match_intervel[0], match_intervel[1]]
                # local_running_task.append(tmptask)
                local_running_answers.append(tmpanswer)

                # reset the local ava time
                reset_avatime_based_anslist(local_ava_time, tmpanswer, match_intervel[0],
                                            match_intervel[1])

                heft_local_answer_list[match_task_time[0]].append([tmptask, match_task_time[1],
                                                                    match_task_time[2]])

                local_aft_list[tmptask] = match_task_time[2]

                tmpsubtask_index += 1
            else:
                break

        reset_avatime_answerlist_aft(ava_time_nlookback, answer_list, aft_list, local_running_answers)

        # running_task += local_running_task
        lookback_start_time += n
        lookback_end_time += n
    return answer_list

def NSGA_n_look_back(n):

    NSGA_offline_ans = NSGA_offline()
    # copy the ava_time_list
    ava_time_nlookback = [[] for i in range(len(ava_time_list))]
    for i in range(len(ava_time_list)):
        for j in range(len(ava_time_list[i])):
            ava_time_nlookback[i].append(ava_time_list[i][j].copy())

    # init parameters
    # running_task = []  # has running task in the look back time
    visited_order = first_order_sequence(pre,succ)  # the first order visited orderf
    aft_list = [-1 for i in range(len(workload))]  # the actual finish time for each task
    answer_list = [[] for i in range(len(ava_time_list))]  # the total answer list

    # for 0 ~ n, get
    lookback_start_time = 0
    lookback_end_time = n
    tmpsubtask_index = 0

    while tmpsubtask_index != len(workload)-1:
        # reset time interval
        local_ava_time = [[] for i in range(len(ava_time_list))]
        for i in range(len(ava_time_list)):
            for tmpavatime in ava_time_nlookback[i]:
                if tmpavatime[1] <= lookback_end_time and tmpavatime[0] >= lookback_start_time:
                    local_ava_time[i].append(tmpavatime.copy())
                else:
                    if tmpavatime[0] < lookback_end_time and tmpavatime[0] >= lookback_start_time:
                        if tmpavatime[0] - lookback_end_time != 0:
                            local_ava_time[i].append([tmpavatime[0], lookback_end_time])

                    if tmpavatime[0] < lookback_start_time:
                        if tmpavatime[1] > lookback_start_time and tmpavatime[1] <= lookback_end_time:
                            if tmpavatime[0] - lookback_end_time != 0:
                                local_ava_time[i].append([lookback_start_time, tmpavatime[1]])
                    if tmpavatime[0] < lookback_start_time and tmpavatime[1] > lookback_end_time:
                        local_ava_time[i].append([lookback_start_time, lookback_end_time])
        # local running task
        # local_running_task = []
        # local running answer
        local_running_answers = []
        # local task finish time
        local_aft_list = [tmp for tmp in aft_list]

        # heft answer list init with copy with answer list
        heft_local_answer_list = [[] for i in range(len(ava_time_list))]
        for i in range(len(answer_list)):
            for tmptimeinterval in answer_list[i]:
                heft_local_answer_list[i].append(tmptimeinterval.copy())


        # choose the tasks from unfinish task and get heft offline answer, task finish time
        while tmpsubtask_index != len(workload)-1:
            
            tmptask = visited_order[tmpsubtask_index]

            if tmptask == 0 or tmptask == len(visited_order)-1:
                tmpsubtask_index += 1
                continue

            tmptaskans = get_offline_answer(NSGA_offline_ans, tmptask)

            ava_timw_now = local_ava_time[tmptaskans[1]]

            if len(ava_timw_now) == 0:
                break

            if len(pre[tmptask]) == 0:
                est = 0
            elif pre[tmptask][0] == 0:
                est = ava_timw_now[0][0]
            else:
                est = max([local_aft_list[j] + computer_Cij(heft_local_answer_list, j, tmptask, tmptaskans[1])
                            for j in pre[tmptask]])

            eft = est + workload[tmptask] / edge_computer_capability[tmptaskans[1]]

            # match with local ava time list
            match_task_time, match_intervel = match_time_interval(est, eft, local_ava_time, tmptaskans[1],
                                                                    tmptask)

            if match_task_time != None:
                tmpanswer = [tmptask, match_task_time[0], match_task_time[1],
                                match_task_time[2], match_intervel[0], match_intervel[1]]
                # local_running_task.append(tmptask)
                local_running_answers.append(tmpanswer)

                # reset the local ava time
                reset_avatime_based_anslist(local_ava_time, tmpanswer, match_intervel[0],
                                            match_intervel[1])

                heft_local_answer_list[match_task_time[0]].append([tmptask, match_task_time[1],
                                                                    match_task_time[2]])

                local_aft_list[tmptask] = match_task_time[2]

                tmpsubtask_index += 1
            else:
                break

        reset_avatime_answerlist_aft(ava_time_nlookback, answer_list, aft_list, local_running_answers)

        # running_task += local_running_task
        lookback_start_time += n
        lookback_end_time += n
    return answer_list

# 获取算法结果的完成时间
def get_max_time(anslist):
    max_time = -1
    if anslist == 2 or anslist == 3:
        return max_time
    for tmp in anslist:
        tmp.sort(key=lambda x: x[2], reverse=True)

        if len(tmp) != 0:
            max_time = max(max_time, tmp[0][2])

    return max_time

if __name__ == "__main__":

    # connect_graph()
    # print(pre)
    # print(succ)
    # print(workload)
    # print(datasize)
    visit_order = first_order_sequence(pre,succ)
    print(visit_order)

    # print(taskindex2order_map)
    # print(order2taskindex_map)
    
    # print(visit_order)

    # print(decision_time_list)
    # print(ava_time_list)

    # greedy_time_reservation_ans_list = greedy_time_reservation()
    # # print(greedy_time_reservation_ans_list)
    # print(get_max_time(greedy_time_reservation_ans_list))
    # # plot_result(greedy_time_reservation_ans_list, ava_time_list, title="Greedy-Reservation Actual Time")
    

    # greedy_ans_list = greedy()
    # # print(greedy_ans_list)
    # print(get_max_time(greedy_ans_list))
    # # plot_result(greedy_ans_list, ava_time_list, title="Greedy Actual Time")
    
    # lbck_ans_list = lbck()
    # print(lbck_ans_list)
    # plot_result(lbck_ans_list, ava_time_list, title="LBCK Schedule Ansers")

    # heft_time_reservation_actual_ans_list = heft_time_reservation()
    # print(heft_time_reservation_actual_ans_list)
    # # plot_result(theory_ans_list, ava_time_list, title="HEFT Theoretical Time")
    # plot_result(heft_time_reservation_actual_ans_list, ava_time_list, title="HEFT-Reservation Actual Time")

    # heft_actual_ans_list = heft()
    # # print(heft_actual_ans_list)
    # print(get_max_time(heft_actual_ans_list))
    # # plot_result(theory_ans_list, ava_time_list, title="HEFT Theoretical Time")
    # # plot_result(heft_actual_ans_list, ava_time_list, title="HEFT Actual Time")

    greedy_nlook_back_ans = greedy_nlook_back(window_size)
    # print(greedy_nlook_back_ans)
    print(get_max_time(greedy_nlook_back_ans))
    # plot_result(greedy_nlook_back_ans, ava_time_list, title="Greedy_nlook Time")

    heft_n_look_back_ans = heft_n_look_back(window_size)
    # print(heft_n_look_back_ans)
    print(get_max_time(heft_n_look_back_ans))
    # plot_result(heft_n_look_back_ans, ava_time_list, title="HEFT_nlook Time")

    greedy_time_reservation_nlook_back_ans = greedy_time_reservation_nlook_back(window_size)
    # print(greedy_time_reservation_nlook_back_ans)
    print(get_max_time(greedy_time_reservation_nlook_back_ans))
    # plot_result(greedy_time_reservation_nlook_back_ans, ava_time_list, title="Gready_reservation_nlook Time")

    # heft_n_look_back_Improved_ans = heft_n_look_back_improved(window_size)
    # # print(heft_n_look_back_ans)
    # print(get_max_time(heft_n_look_back_Improved_ans))
    # # plot_result(heft_n_look_back_ans, ava_time_list, title="HEFT_nlook_Improved Time")


    # answear_nsga = NSGA()
    # print(get_max_time(answear_nsga))
    # plot_result(answear_nsga, ava_time_list, title="Gready_reservation_nlook Time")

    NSGA_n_look_back_ans = NSGA_n_look_back(window_size)
    print(get_max_time(NSGA_n_look_back_ans))
    # plot_result(NSGA_n_look_back_ans, ava_time_list, title="NSGA_nlook Time")

    # print(get_algorithm_timelist())
    