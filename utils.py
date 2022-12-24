import numpy as np
from operator import itemgetter

def get_layer_temps(temp_balance,n_alphas,epoch_val):
    n = len(n_alphas)
    idx = [i for i in range(n)]
    temps = np.array([epoch_val] * n)

    if temp_balance == 'tbr':
        idx = np.argsort(n_alphas)
        temps = [2 * epoch_val * (0.35 + 0.15 * 2 * i / n) for i in range(n)]
    elif temp_balance == 'avg':
        temps = n_alphas/np.sum(n_alphas) * n * epoch_val
    elif temp_balance == 'sqrt':
        temps = np.sqrt(n_alphas)/np.sum(np.sqrt(n_alphas)) * n * epoch_val
    elif temp_balance == 'cbrt':
        temps = np.cbrt(n_alphas)/np.sum(np.cbrt(n_alphas)) * n * epoch_val
    elif temp_balance == 'log2':
        temps = np.log2(n_alphas)/np.sum(np.log2(n_alphas)) * n * epoch_val
    elif temp_balance == 'square':
        temps = np.power(n_alphas,2)/sum(np.power(n_alphas,2)) * n * epoch_val
    elif temp_balance == 'negpow2':
        idx = np.argsort(n_alphas)
        neg_pow2 = np.exp2(range(-n,0))
        neg_pow2[-1] = neg_pow2[-1] + (1-np.sum(neg_pow2))
        temps = neg_pow2  * n * epoch_val
    elif temp_balance == 'softmax':
        temps = np.exp(n_alphas)/np.sum(np.exp(n_alphas)) * n * epoch_val
    elif temp_balance == 'sample_lr':
        idx = np.argsort(n_alphas)
        samples = np.sort(np.random.normal(epoch_val,epoch_val/2,n))
        # shift to make nonnegative
        samples = samples - min(samples)
        samples_prob = np.divide(samples,np.sum(samples))
        temps = samples_prob * n * epoch_val
    elif temp_balance == 'sample_alpha':
        samples = []
        for i in range(n):
            samples.append(np.random.normal(n_alphas[i],1,1).item())
        samples=np.array(samples)
        samples = samples - min(samples)
        samples_prob = np.divide(samples,np.sum(samples))
        temps = samples_prob * n * epoch_val
    elif temp_balance == 'sample_AWGN':
        idx = np.argsort(n_alphas)
        samples = np.sort(np.random.normal(np.mean(n_alphas),np.sqrt(1+np.var(n_alphas)),n))
        samples = samples - min(samples)
        samples_prob = np.divide(samples,np.sum(samples))
        temps = samples_prob * n * epoch_val
    return [value for index, value in sorted(list(zip(idx, temps)), key=itemgetter(0))]