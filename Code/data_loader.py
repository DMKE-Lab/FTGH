import random
import numpy as np


class DataLoader(object):
    def __init__(self, dataset, parameter, step='train'):
        self.curr_rel_idx = 0
        self.tasks = dataset[step+'_tasks']
        self.rel2candidates = dataset['rel2candidates']
        self.e1rel_e2 = dataset['e1rel_e2']
        self.all_rels = sorted(list(self.tasks.keys()))
        self.num_rels = len(self.all_rels)
        self.few = parameter['few']
        self.bs = parameter['batch_size']
        self.nq = parameter['num_query']

        if step != 'train':
            self.eval_quadruples = []
            for rel in self.all_rels:
                self.eval_quadruples.extend(self.tasks[rel][self.few:])
            self.num_tris = len(self.eval_quadruples)
            self.curr_tri_idx = 0

    def next_one(self):
        # 将curr_rel_idx移至所有关系的一个循环后的0
        if self.curr_rel_idx % self.num_rels == 0:
            random.shuffle(self.all_rels)
            self.curr_rel_idx = 0

        # 获取当前关系和当前候选人
        curr_rel = self.all_rels[self.curr_rel_idx]
        self.curr_rel_idx = (self.curr_rel_idx + 1) % self.num_rels  # shift current relation idx to next
        curr_cand = self.rel2candidates[curr_rel]
        while len(curr_cand) <= 10 or len(self.tasks[curr_rel]) <= 10:  # ignore the small task sets
            curr_rel = self.all_rels[self.curr_rel_idx]
            self.curr_rel_idx = (self.curr_rel_idx + 1) % self.num_rels
            curr_cand = self.rel2candidates[curr_rel]

        # 通过curr_rel从所有任务中获取当前任务并对其进行无序处理
        curr_tasks = self.tasks[curr_rel]
        curr_tasks_idx = np.arange(0, len(curr_tasks), 1)
        curr_tasks_idx = np.random.choice(curr_tasks_idx, self.few+self.nq)
        support_quadruples = [curr_tasks[i] for i in curr_tasks_idx[:self.few]]
        query_quadruples = [curr_tasks[i] for i in curr_tasks_idx[self.few:]]

        # 构造支持和查询负四元组
        support_negative_quadruple = []
        for quadruple in support_quadruples:
            e1, rel, e2, t = quadruple
            while True:
                negative = random.choice(curr_cand)
                if (negative not in self.e1rel_e2[e1 + rel+t]) \
                        and negative != e2:
                    break
            support_negative_quadruple.append([e1, rel,t, negative])

        negative_quadruple = []
        for quadruple in query_quadruples:
            e1, rel,t,e2 = quadruple
            while True:
                negative = random.choice(curr_cand)
                if (negative not in self.e1rel_e2[e1 + rel+t]) \
                        and negative != e2:
                    break
            negative_quadruple.append([e1, rel,t, negative])

        return support_quadruples, support_negative_quadruple, query_quadruples, negative_quadruple, curr_rel

    def next_batch(self):
        next_batch_all = [self.next_one() for _ in range(self.bs)]

        support, support_negative, query, negative, curr_rel = zip(*next_batch_all)
        return [support, support_negative, query, negative], curr_rel

    def next_one_on_eval(self):
        if self.curr_tri_idx == self.num_tris:
            return "EOT", "EOT"

        # 获取当前四元组
        query_quadruple = self.eval_quadruple[self.curr_tri_idx]
        self.curr_tri_idx += 1
        curr_rel = query_quadruple[1]
        curr_cand = self.rel2candidates[curr_rel]
        curr_task = self.tasks[curr_rel]

        # 获取支持四元组
        support_quadruple = curr_task[:self.few]

        # 构造支持否定
        support_negative_quadruple = []
        shift = 0
        for quadruple in support_quadruple:
            e1, rel, t, e2 = quadruple
            while True:
                negative = curr_cand[shift]
                if (negative not in self.e1rel_e2[e1 + rel]) \
                        and negative != e2:
                    break
                else:
                    shift += 1
            support_negative_quadruple.append([e1, rel, t, negative])

        # 构造负四元组
        negative_quadruples = []
        e1, rel, t, e2 = query_quadruple
        for negative in curr_cand:
            if (negative not in self.e1rel_e2[e1 + rel+t]) \
                    and negative != e2:
                negative_quadruples.append([e1, rel,t, negative])

        support_quadruples = [support_quadruple]
        support_negative_quadruple = [support_negative_quadruple]
        query_quadruple = [[query_quadruple]]
        negative_quadruples = [negative_quadruples]

        return [support_quadruples, support_negative_quadruple, query_quadruple, negative_quadruples], curr_rel

    def next_one_on_eval_by_relation(self, curr_rel):
        if self.curr_tri_idx == len(self.tasks[curr_rel][self.few:]):
            self.curr_tri_idx = 0
            return "EOT", "EOT"

        # 获取当前四元组
        query_quadruple = self.tasks[curr_rel][self.few:][self.curr_tri_idx]
        self.curr_tri_idx += 1
        # curr_rel = query_quadruple[1]
        curr_cand = self.rel2candidates[curr_rel]
        curr_task = self.tasks[curr_rel]

        # 获取支持四元组
        support_quadruples = curr_task[:self.few]

        #构造支持否定
        support_negative_quadruples = []
        shift = 0
        for quadruple in support_quadruples:
            e1, rel, t,e2 = quadruple
            while True:
                negative = curr_cand[shift]
                if (negative not in self.e1rel_e2[e1 + rel+t]) \
                        and negative != e2:
                    break
                else:
                    shift += 1
            support_negative_quadruples.append([e1, rel, t, negative])

        # 构造负三元组
        negative_quadruples = []
        e1, rel,t,e2 = query_quadruple
        for negative in curr_cand:
            if (negative not in self.e1rel_e2[e1 + rel]) \
                    and negative != e2:
                negative_quadruples.append([e1, rel, negative])

        support_quadruples = [support_quadruples]
        support_negative_quadruples = [support_negative_quadruples]
        query_quadruple = [[query_quadruple]]
        negative_quadruples = [negative_quadruples]

        return [support_quadruples, support_negative_quadruples, query_quadruple, negative_quadruples], curr_rel

