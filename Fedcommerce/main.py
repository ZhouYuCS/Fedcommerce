import numpy as np
import random
import os
from datetime import datetime
import math
import pandas as pd
seed = 6
random.seed(seed)  # 设置随机种子
np.random.seed(seed)
from Simulater import *
import torch
from datetime import datetime
import copy
import json
class WeightedMajorityAuction:

    def __init__(self, candidate_prices, eta, h):
   
        self.candidate_prices = candidate_prices
        self.weights = {
            price: 1.0 / len(self.candidate_prices)
            for price in candidate_prices
        }  # Initialize weights to 1
        self.eta = eta
        self.MP_profit=-1
        self.history_bids = []
        self.history_alpha_wm=0
        self.history_alpha_mp=0
        self.h = h
        self.history_profits=0
        self.remain_time=0

    def get_price(self, bid):
        chosen_price = random.choices(
            self.candidate_prices,
            weights=[self.weights[price]
                     for price in self.candidate_prices])[0]
        self.history_bids.append(bid)
        # Update weights based on the bid
        for price in self.candidate_prices:
            if price <= self.history_bids[
                    -1]:  # If the bid is higher than the price
                self.weights[price] *= math.pow(1 + self.eta, price / self.h)

        # Normalize weights
        total_weight = sum(self.weights.values())
        for price in self.candidate_prices:
            self.weights[price] /= total_weight
        
        return chosen_price
    

class Server:

    def __init__(
        self,
        N_DO,
        N_DC,
        Dur,
        tau,
        h,
        l,
        N_task,
        categories,
        the,
        punishment,
        eta,
        Model_family,
        base_model,
        Val,
        buffer,
        rg,
        fs
    ):
        self.N_DO = N_DO
        self.N_DC = N_DC
        self.Dur = Dur
        self.tau = tau
        self.h=h
        self.l=l
        self.N_task = N_task
        self.categories = categories

        self.pin_MP_flag=True
        self.pin_WM_flag=True
        self.pin_MP=[N_task*N_DO-3,N_task*N_DO-2]    # MP 实验的商品编号,自己定
        self.pin_WM=[]      # WM 实验最后一次实验的时间，后两个
        self.downloads={i:0 for i in range(self.N_DO,self.N_DO+self.N_DC)}
        self.the = the
        self.punishment = punishment
        self.eta = eta

        self.Model_family = Model_family
        self.base_model = base_model
        self.Val = Val
        self.buffer=buffer
        self.rg=rg
        self.fs=fs
        self.forbidden={i:set() for i in range(self.N_DO,self.N_DO+self.N_DC)}
        self.uploaded_number_so_far = 0
        self.aggregation_time_WM=0
        self.aggregation_time_MP=0
        
        self.rectangle2line = [[] for i in range(self.N_DO)
                               ]  #from [DO,task] to model index
        self.line2rectangle = {}
        self.WM_cache = {
        }  #model index to WeightedMajorityAuction, for MP pro calculation
        self.acc_vector_cache = {}  # elementary model index to acc vector
        self.acc_vector_cache_temp = {}  # temprory model index to acc vector
        self.elementary_model_base = {
        }  # keys are index of elementary model, values are entity
        self.temporary_model_base = {
        }  # keys are indexes od DC, values are model entities
        self.pr_table = {i:[] for i in range(self.N_DO,self.N_DO+self.N_DC)}  # dc's purchase records
        self.bi_table = {}  # dc's current bo's (b,w)
        self.bidding_set = [1 + i * (h - 1) / (l - 1) for i in range(l)]
        
        self.MP_bin_price={i:0 for i in self.bidding_set}
        self.MP_number_price={i:0 for i in self.bidding_set}
        self.MP_bin_DC ={i:0 for i in range(self.N_DO,self.N_DO+self.N_DC)}
        self.WM_bin_price={i:0 for i in self.bidding_set}
        self.WM_number_price={i:0 for i in self.bidding_set}
        self.WM_bin_DC={i:0 for i in range(self.N_DO,self.N_DO+self.N_DC)}

        self.events = []
        self.simulate_model_events()
        self.events.sort(key=lambda x: x[0])
        self.DO_bills = {}  # tuple denotes (money,commodity index)
        self.DC_bills = {}

        for i in range(len(self.events)):
            time, content, who, kind = self.events[i]
            self.rectangle2line[who].append(i)
        
        for i in range(len(self.rectangle2line)):
            for j in range(len(self.rectangle2line[i])):
                self.line2rectangle[self.rectangle2line[i][j]]=(i,j)

        self.simulate_bid_events()
        self.events.sort(key=lambda x: x[0])
        WM_num=10
        for i in range(len(self.events)):
            event=self.events[-(i+1)]
            time, content, who, kind = event
            if kind=='be':
                self.pin_WM.append(time)
                WM_num-=1
            if WM_num==0:
                break

        self.FL_acc_vector=self.get_acc_vector(self.base_model)

    def get_acc_vector(self,model_pair): #return numpy list
        correct_counts = torch.zeros(50+self.categories,dtype=int)
        total_counts = torch.zeros(50+self.categories, dtype=int)
        W=RI(model_pair[0],model_pair[1],model_pair[3],self.rg,self.fs)
        #W=model_pair[0]
        with torch.no_grad():  # 不计算梯度
            for images, labels in self.Val:
                images = self.buffer(images)
                outputs = images @ W
                _, predicted = torch.max(outputs, 1)
                for label, pred in zip(labels, predicted):
                    total_counts[label] += 1
                    if label == pred:
                        correct_counts[label] += 1
        accuracies = (correct_counts / total_counts).numpy()
        return accuracies 
        #return np.random.rand(50+self.categories)

    def simulate_model_events(self):
        for i in range(self.N_DO):
            TR_junctures = self.simulate_junctures(self.N_task)
            for j in range(len(TR_junctures)):
                self.events.append(
                    (TR_junctures[j], self.Model_family[i][j], i, 'me'))

    def simulate_bid_events(self):
        for i in range(self.N_DO,self.N_DO+self.N_DC):
            if  self.punishment>0 and i>self.N_DC-10:
                bid_index=self.N_DC-i
                times = random.randint(1, self.N_task)
                BO_junctures = self.simulate_junctures(times)
                for BO_juncture in BO_junctures:
                    BO_value = self.bidding_set[-bid_index]
                    BO_weight = self.simulate_BO_weight()
                    self.events.append(
                        (BO_juncture, (BO_value, BO_weight), i, 'be'))
            else:
                times = random.randint(1, self.N_task)
                BO_junctures = self.simulate_junctures(times)
                for BO_juncture in BO_junctures:
                    BO_value = self.simulate_BO_value()
                    BO_weight = self.simulate_BO_weight()
                    self.events.append(
                        (BO_juncture, (BO_value, BO_weight), i, 'be'))
        

    def simulate_BO_value(self):
        bid = random.choice(self.bidding_set)
        return bid

    def simulate_BO_weight(self):#mu=0.1
        weights = np.random.dirichlet(np.ones(self.categories)*0.1,
                                      size=1).flatten()
        pre=np.zeros(50)*0.1
        weights=np.concatenate((pre,weights))
        return weights

    def simulate_junctures(self, times):
        intervals = np.random.dirichlet(np.ones(times + 1),
                                        size=1).flatten() * self.Dur
        return np.cumsum(intervals).tolist()[:-1]
    
    #WM
    def bid_event(self, time, bidding, demand_weights, DC_index,log_file):
        
        self.forbidden[DC_index]={t for t in self.forbidden[DC_index] if t[1]>time}
        self.bi_table[DC_index]=(bidding,demand_weights)
        #construct initial model
        purchased_indexes = self.pr_table[DC_index]
        if len(purchased_indexes)>0:
            initial_model,last_acc_vector = self.generate_initial_model(
                demand_weights, purchased_indexes)
        else:
            initial_model = self.base_model
            last_acc_vector =  self.FL_acc_vector
        #recursilvly buy
        np_demand = np.array(demand_weights)
        payments = {}
        sorted_alphas = sorted(self.acc_vector_cache.items(),
                               key=lambda item: np.array(item[1]) @ np_demand,
                               reverse=True)
        if self.pin_WM_flag and time in self.pin_WM:
            WM_file = self.wm_path+'/'+str(time)+'.log'
            wm_exp= open(WM_file, 'w')
            wm_exp.write('model_id,best_profits,real_profits\n')

        for i, (index, model) in enumerate(sorted_alphas):
            if index not in purchased_indexes and (index,) not in self.forbidden[DC_index]:
                model = self.elementary_model_base[index]
                aggregated_model=aggregate(initial_model,model)
                self.aggregation_time_WM+=1
                new_acc_vector=self.get_acc_vector(aggregated_model)
                metric=new_acc_vector@np_demand
                true_alpha =  metric-last_acc_vector@np_demand
                if true_alpha>self.the:
                    WMA=self.WM_cache[index]
                    WMA.history_alpha_wm+=true_alpha
                    price,flag=self.WM(bidding,WMA)
                    if self.pin_WM_flag and time in self.pin_WM:
                        bids=WMA.history_bids
                        sorted_bids = sorted(bids, reverse=True)
                        max_revenue = 0
                        optimal_fee = 0
                        for j in range(len(sorted_bids)):
                            revenue = (j + 1) * sorted_bids[j]
                            if revenue >= max_revenue:
                                max_revenue = revenue
                                optimal_fee = sorted_bids[j]
                        wm_exp.write(str(index)+','+str(max_revenue)+','+str(WMA.history_profits)+'\n')
                    if flag:
                       
                        self.WM_bin_DC[DC_index]+=1
                        self.WM_bin_price[bidding]+=1
                        payments[index]=price*true_alpha
                        last_acc_vector=new_acc_vector
                        initial_model=aggregated_model            
                        log_file.write(str(index)+ ',' + str(DC_index) + ',' + str(self.line2rectangle[index][0]) + ',succ,' + str(bidding) + ',' + str(true_alpha) + ',' + str(price) + ',' + str(payments[index]) + ',' + str(metric)+','+str(time) + ',b\n')
                    else:
                        self.forbidden[DC_index].add((index,self.punishment+time))
                        log_file.write(str(index)+ ',' + str(DC_index) + ',' + str(self.line2rectangle[index][0]) + ',fail,' + str(bidding) + ',' + str(true_alpha) + ',' + str(price) + ',' + str(0) + ',' + str(metric)+','+str(time) + ',b\n')
                    self.WM_number_price[bidding]+=1
        new_purchased_models=list(payments.keys())
        if len(new_purchased_models)>0:
            self.pr_table[DC_index].extend(new_purchased_models)
            self.downloads[DC_index]+=1
        self.temporary_model_base[DC_index]=initial_model
        self.acc_vector_cache_temp[DC_index]=last_acc_vector

    #MP
    def model_event(self, time, new_model, who,log_file):
        new_id=len(self.elementary_model_base)
        self.elementary_model_base[new_id]=new_model
        new_acc_vector=self.get_acc_vector(new_model)
        self.acc_vector_cache[new_id]=new_acc_vector
        self.WM_cache[new_id]= WeightedMajorityAuction(self.bidding_set,self.eta,self.h)
        self.WM_cache[new_id].remain_time=self.Dur-time
        competers={}
        precache_models={}
        precache_alphas={}
        metrics={}
        cache_vector={}
        # 判断 alpha
       
        for bo_index, wb_pair in self.bi_table.items():
    # 现在您可以使用 key, bo_index, 和 wb_pair
            bidding=wb_pair[0]
            demand_weights=wb_pair[1]
            initial_model=self.temporary_model_base[bo_index]
            aggeregate_model=aggregate(initial_model,new_model)
            self.aggregation_time_MP+=1
            cache_acc_vector=self.get_acc_vector(aggeregate_model)
            metric=np.array(demand_weights)@cache_acc_vector
            alpha=metric-np.array(demand_weights)@self.acc_vector_cache_temp[bo_index]
            alpha_sum=0
            if alpha>self.the:
                competers[bo_index]=bidding
                precache_models[bo_index]=aggeregate_model
                precache_alphas[bo_index]=alpha
                metrics[bo_index]=metric
                cache_vector[bo_index]= cache_acc_vector
                alpha_sum+=alpha

        if len(competers)>0:
            optimal_fee, winners= self.MP(competers)
            if self.pin_MP_flag and new_id in self.pin_MP:
                MP_file = self.mp_path+'/'+str(new_id)+'.log'
                mp_exp= open(MP_file, 'w')
                mp_exp.write('cheater,original_price,fakebid,fakeprice,fakeutility\n')
            
                for cheater,original_price in competers.items():
                    fake_competers=copy.deepcopy(competers)
                    for fakebid in self.bidding_set:
                        fake_competers[cheater]=fakebid
                        optimal_fee_fake, winner_fake= self.MP(fake_competers)
                        if cheater in winner_fake:
                            mp_exp.write(str(cheater)+','+str(original_price)+','+str(fakebid)+','+str(optimal_fee_fake)+','+str(original_price-optimal_fee_fake)+'\n')
                        else:
                            mp_exp.write(str(cheater)+','+str(original_price)+','+str(fakebid)+','+str(optimal_fee_fake)+','+str(0)+'\n')      
            #存储
            for bo_index, wb_pair in self.bi_table.items():
                if bo_index in winners:
                    self.MP_number_price[wb_pair[0]]+=1
                    self.pr_table[bo_index].append(new_id)
                    self.temporary_model_base[bo_index]=precache_models[bo_index]
                    self.acc_vector_cache_temp[bo_index]=cache_vector[bo_index]
                    self.MP_bin_price[wb_pair[0]]+=1
                    self.MP_bin_DC[bo_index]+=1
                    self.downloads[bo_index]+=1
                    log_file.write(str(new_id)+ ',' + str(bo_index) + ',' + str(who) + ',succ,' + str(self.bi_table[bo_index][0]) + ',' + str(precache_alphas[bo_index]) + ',' + str(optimal_fee) + ',' + str(precache_alphas[bo_index]*optimal_fee) + ',' + str(metrics[bo_index])+','+str(time) + ',m\n')
                elif bo_index in competers.keys():
                    self.MP_number_price[wb_pair[0]]+=1
                    log_file.write(str(new_id)+ ',' + str(bo_index) + ',' + str(who) + ',fail,' + str(self.bi_table[bo_index][0]) + ',' + str(precache_alphas[bo_index]) + ',' + str(optimal_fee) + ',' + str(0) + ',' + str(metrics[bo_index])+','+str(time) + ',m\n')
            self.WM_cache[new_id].MP_profit= optimal_fee*len(winners)
            self.WM_cache[new_id].history_alpha_mp=alpha_sum

    def WM(self, bid, WMA):
        price = WMA.get_price(bid)
        if bid < price:    
            return price,False
        else:
            WMA.history_profits+=price
            return price,True

    def MP(self, bids):
        sorted_bids = sorted(bids.items(),
                             key=lambda item: item[1],
                             reverse=True)
        max_revenue = 0
        optimal_fee = 0
        winners = []
        users=[]
        for i, (user, bid) in enumerate(sorted_bids):
            revenue = (i + 1) * bid
            users.append(user)
            if revenue >= max_revenue:
                max_revenue = revenue
                optimal_fee = bid
                winners=copy.deepcopy(users)
        return optimal_fee, winners
        
    

    def generate_initial_model(self, demands, purchased_indexes):
        initial_model = self.base_model
        last_acc_vector=self.FL_acc_vector
        np_demand = np.array(demands)
        sorted_alphas = sorted(self.acc_vector_cache.items(),
                               key=lambda item: np.array(item[1]) @ np_demand,
                               reverse=True)
        for i, (index, model) in enumerate(sorted_alphas):
            if index in purchased_indexes:
                model = self.elementary_model_base[index]
                aggregated_model=aggregate(initial_model,model)
                self.aggregation_time_WM+=1
                new_acc_vector=self.get_acc_vector(aggregated_model)
                metric=new_acc_vector@np_demand
                true_alpha =  metric-last_acc_vector@np_demand
                if true_alpha>0:
                    initial_model=aggregated_model  
                    last_acc_vector= new_acc_vector  
        return initial_model,last_acc_vector
    
   

    def boost(self,data_name,args):
     
        now = datetime.now()
        timestamp = now.strftime('%Y%m%d_%H%M%S')
        self.dir_name=f'./{timestamp}_'+data_name+'_'+str(args["N_DO"])+'_'+str(args["N_DC"])+'_'+str(args["N_task"])+'_'+str(args["Omega"])
        os.makedirs(self.dir_name)
      
        log_file_name_BO = self.dir_name+'/BO.log'
        log_file_BO= open(log_file_name_BO, 'w')
        log_file_BO.write('time,who,bidding\n')
        pd_list=[]
        for event in self.events:
            time, content, who, kind = event
            if kind=='be':
                 base=content[1][50:].tolist()
                 base.append(time)
                 pd_list.append(base)
                 log_file_BO.write(str(time)+ ',' + str(who) + ',' + str(content[0]) + '\n')
        with open(self.dir_name+'/weights.json','w') as f:
            json.dump(pd_list, f)
        

        log_file_name = self.dir_name+'/transactions.log'
        log_file = open(log_file_name, 'w')
        log_file.write('model,dc,do,suc,b,alpha,price,payment,metric,time,kind\n')
        
        
       
        self.mp_path=self.dir_name+'/'+'MP'
        self.wm_path=self.dir_name+'/'+'WM'
        os.makedirs(self.mp_path)
        os.makedirs(self.wm_path)

        start_time = datetime.now()
        for event in self.events:
            time, content, who, kind = event
            if kind == "be":
                value, weight = content
                self.bid_event(time, value, weight, who,log_file)
            else:
                self.model_event(time, content, who,log_file)
        end_time = datetime.now()
        print(f"Execution time: {end_time - start_time}")
        print(self.FL_acc_vector)
        print(self.aggregation_time_WM)
        print(self.aggregation_time_MP)
        pd_list=[]
        for key,value in  self.acc_vector_cache_temp.items():
           base=value.tolist()
           base.append(time)
           pd_list.append(base)
        with open(self.dir_name+'/finalacc.json','w') as f:
            json.dump(pd_list, f)
        

    def test(self):
        print("-----------------------")
        for key,value in self.WM_cache.items():
            print(value.history_bids)
        for i in range(self.N_DO,self.N_DC+self.N_DO):
            print(self.pr_table[i])
        
    def testfor22(self):
        print("-----------------------testfor22")
        with open(self.dir_name+'/testfor22_MP.json', 'w') as f:
            json.dump(self.MP_bin_DC, f)
        with open(self.dir_name+'/testfor22_WM.json', 'w') as f:
            json.dump(self.WM_bin_DC, f)

   
    def testfor17(self):
        print("-----------------------testfor17")
        dict=[self.WM_bin_price,self.WM_number_price,self.MP_bin_price,self.MP_number_price]
        with open(self.dir_name+'/test17.json', 'w') as f:
            json.dump(dict, f)
       
   
       
    
    def test21(self):
        print("-----------------------test21")
        '''
        for key,value in self.downloads.items():
            print(key)
            print(value)
        '''
        with open(self.dir_name+'/test21.json', 'w') as f:
            json.dump(self.downloads, f)
    
    def test16(self):
        print("-----------------------test16")
        dict_16={}
        for key,value in self.WM_cache.items():
            dict_16[key]={'history_alpha_wm':value.history_alpha_wm,'history_alpha_mp':value.history_alpha_mp,'ratio':self.elementary_model_base[key][-1],'remain_time':value.remain_time,}
        with open(self.dir_name+'/test16.json', 'w') as f:
            json.dump(dict_16, f)

          

def main(agrs):
    data_name=args['data_name']
    buffer_size =args['buffer_size']
    rg = 0.1
    a = args["N_DO"]
    class_number = 100 if     data_name=='C' else 85
    args["categories"]=50 if  data_name=='C' else 35
    K = args["N_task"]

    ratio = [1.0]  #每个 DO怎么分数据量
    ratio = np.random.dirichlet(np.ones(a), size=1).flatten().tolist()#Omega

    partition_sizes = [[1 / K] * K] * a  #在每一个DO内部怎么分数据量需要协调，从 dir 中采样
    partition_sizes = [[] for i in range(a)]
    for i in range(a):
        partition_sizes[i] = np.random.dirichlet(np.ones(K), #omega
                                                 size=1).flatten().tolist()

    partition_alphas = [args['Omega']] * a  # 每个 DO 内的 alpha，【0，2】之间
  
    # DO之间的aplha

    re_align_loader, server_loader, divided_loaders, feature_size = prepare_dataloader(
        a, ratio, K, partition_sizes, partition_alphas, seed,data_name)

    buffer = RandomBuffer(
        feature_size,
        buffer_size,
    )
    base_tuple = AFL(re_align_loader, class_number, buffer, rg, buffer_size,1)
    do_task_to_tuple = {key: dict() for key in [i for i in range(a)]}

    for do_index in range(a):
        for task_index in range(K):
            do_task_to_tuple[do_index][task_index] = AFL(
                divided_loaders[do_index][task_index], class_number, buffer,
                rg, buffer_size,partition_sizes[do_index][task_index]*ratio[do_index])
   

    server = Server(N_DO=args["N_DO"],
                    N_DC=args["N_DC"],
                    Dur=args["Dur"],
                    tau=args["tau"],
                    h=agrs["h"],
                    l=args["l"],
                    N_task=args["N_task"],
                    categories=agrs["categories"],
                    the=args["the"],
                    punishment=args["punishment"],
                    eta=args["eta"],
                    Model_family= do_task_to_tuple ,
                    base_model=base_tuple,
                    Val=server_loader,
                    buffer=buffer,
                    rg=rg,
                    fs=buffer_size)

    server.boost(data_name,args)
   
    server.test16()
    server.test21()
    server.testfor17()
    server.testfor22()

if __name__ == "__main__":
    Omega=1
    condition=[
         [25,'C',  Omega],
             [50,'C',  Omega],
              [75,'C',  Omega],
               [100,'C',  Omega],
        [25,'I',  Omega],
         [50,'I',  Omega],
          [75,'I',  Omega],
           [100,'I',  Omega],       
    ]
  
    condition=[
            
               [75,'C',  0.1],
    ]
    for script in condition:
        if script[1]=='C':
            DO=5
        else:
            DO=4
        args = {
            "N_DO": 5,
            "N_DC": script[0],           #25 50 75 100
            "Dur": 200,
            "tau": 10,
            "h": 101,
            "l": 21,
            "N_task": 3,
            "categories": 50,
            "the": 0.005,
            "punishment": 0,
            "eta": 0.5,
            "buffer_size":2048,
            "data_name":script[1],
            "Omega":script[2]
        }
        main(args)
