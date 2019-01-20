import torch.optim.lr_scheduler as lr_scheduler
import math 

class GradientRatioScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, last_epoch=-1, smooth=0):
        self.lr_factors = [1 for _ in optimizer.param_groups]
        self.smooth = smooth
        super(GradientRatioScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.lr_factors[i] for i,base_lr in enumerate(self.base_lrs)]

    def on_after_batch(self):
        this_g_sum, this_g_count = 0, 0
        last_g = None

        i, m_i = 0, 0
        i_start = None
        pgs = list(self.optimizer.param_groups)
        while i < len(pgs):
            pg = pgs[i]
            #TODO remove dep on m_i?
            if pg['m_i'] != m_i: # new module started
                # save last moddule's avg
                if last_g is None:
                    last_g = this_g_sum / this_g_count
                this_g_sum, this_g_count = 0, 0
                m_i = pg['m_i']
                i_start = i

            # accumulate for current module
            #TODO - worry about requires_grad
            for p in pg['params']:
                this_g_sum += p.grad.abs().sum().item()
                this_g_count += p.grad.numel()
            
            # if this is the last one in module, update LR
            if i_start is not None and (i == len(pgs)-1 or pgs[i+1]['m_i'] != m_i):
                rat = last_g / (this_g_sum / this_g_count)
                if rat < 0.1 or rat > 10 or math.isnan(rat):
                    print(rat, m_i, i)
                for j in range(i, i_start-1, -1):
                    self.lr_factors[j] = \
                        self.smooth*self.lr_factors[j] + (1-self.smooth)*rat
            i += 1