from .algorithm import Algorithm
import math
import numpy as np
import util

class CMCS(Algorithm):
    def value(self, S: np.ndarray):
        l = S.shape[0]
        if l == 0:
            v = self.v_0
        elif l == self.game.n:
            v = self.v_n
        else:
            assert self.func_calls < self.T
            self.func_calls += 1
            v = self.game.value(S)
        return v
    def sample(self, length=None):
        if length is None:
            length = np.random.randint(self.game.n+1)
        S = np.arange(self.game.n)
        np.random.shuffle(S)
        return S[:length]
    def marginal(self, coalition, player, value_coalition=None):
        if value_coalition is None:
            value_coalition = self.value(coalition)
        if player in coalition:
            return value_coalition - self.value(coalition[coalition != player])
        return self.value(np.concatenate((coalition, [player]))) - value_coalition
    def update_phi(self, new_value):
        self.phi = new_value
        self.save_steps(self.step_interval)
        
    def get_top_k(self, k: int, step_interval: int = 100):
        # initialization
        self.step_interval = step_interval
        n = self.game.n
        marginals = np.zeros(n, dtype=np.float32)
        t = np.zeros(n, dtype=np.int32)
        
        # precompute grand and empty coalition values
        self.func_calls += 2
        self.v_n = self.game.value(np.arange(n))
        self.v_0 = self.game.value(np.array([]))
        
        while self.func_calls+2 <= self.T:
            S = self.sample()
            v_S = self.value(S)
            for player in range(n):
                if self.func_calls == self.T:
                    self.update_phi(marginals / t)
                    return
                marginals[player] += self.marginal(S, player, v_S)
                t[player] += 1
            self.update_phi(marginals / t)
        self.func_calls = self.T
        self.save_steps(step_interval)

class Selective_CMCS(CMCS):
    def get_top_k(self, k: int, step_interval: int = 100):
        # initialization
        self.step_interval = step_interval
        n = self.game.n
        marginals = np.zeros(n, dtype=np.float32)
        t = np.zeros(n, dtype=np.int32)
        
        # precompute grand and empty coalition values
        self.func_calls += 2
        self.v_n = self.game.value(np.arange(n))
        self.v_0 = self.game.value(np.array([]))
        
        while self.func_calls+2 <= self.T:            
            # sample players based on uncertainty
            sorted_players = np.argsort(-self.phi)
            # border = np.zeros(n, dtype=np.float32)
            # border[sorted_players[:k]] = self.phi[sorted_players[k]]
            # border[sorted_players[k:]] = self.phi[sorted_players[k-1]]
            border = (self.phi[sorted_players[k-1]] + self.phi[sorted_players[k]])/2
            certainty = np.abs(self.phi - border) * t
            min_certainty, max_certainty = np.min(certainty), np.max(certainty)
            if min_certainty == max_certainty:
                selected_players = np.arange(n)
            else:
                weights = (max_certainty - certainty) / (max_certainty - min_certainty)
                selected_players = np.array((np.random.rand(n) < weights).nonzero())[0]
                
                # if no players where selected, update all to avoid a loop where only a little number of players is selected every time which is not budget efficient 
                if selected_players.shape[0] == 0:
                    selected_players = np.arange(n)
                        
            S = self.sample()
            v_S = self.value(S)
            for player in selected_players:
                if self.func_calls == self.T:
                    self.update_phi(marginals / t)
                    return
                marginals[player] += self.marginal(S, player, v_S)
                t[player] += 1
            self.update_phi(marginals / t)
        self.func_calls = self.T
        self.save_steps(step_interval)

class CMCS_Dependent(CMCS):
    def get_top_k(self, k: int, step_interval: int = 100):
        # initialization
        self.step_interval = step_interval
        n = self.game.n
        marginals = np.zeros(n, dtype=np.float32)
        t = np.zeros(n, dtype=np.int32)
        
        # precompute grand and empty coalition values
        self.func_calls += 2
        self.v_n = self.game.value(np.arange(n))
        self.v_0 = self.game.value(np.array([]))
        
        while self.func_calls+2 <= self.T:
            S = self.sample()
            for player in range(n):
                if self.func_calls+2 > self.T:
                    break
                marginals[player] += self.marginal(S, player)
                t[player] += 1
            self.update_phi(marginals / t)
        self.func_calls = self.T
        self.save_steps(step_interval)
        
class CMCS_Independent(CMCS):
    def get_top_k(self, k: int, step_interval: int = 100):
        # initialization
        self.step_interval = step_interval
        n = self.game.n
        marginals = np.zeros(n, dtype=np.float32)
        t = np.zeros(n, dtype=np.int32)
        
        # precompute grand and empty coalition values
        self.func_calls += 2
        self.v_n = self.game.value(np.arange(n))
        self.v_0 = self.game.value(np.array([]))
        
        while self.func_calls+2 <= self.T:
            for player in range(n):
                if self.func_calls+2 > self.T:
                    break
                S = self.sample()
                marginals[player] += self.marginal(S, player)
                t[player] += 1
            self.update_phi(marginals / t)
            
        self.func_calls = self.T
        self.save_steps(step_interval)
        
class CMCS_Length(CMCS):
    def get_top_k(self, k: int, step_interval: int = 100):
        # initialization
        self.step_interval = step_interval
        n = self.game.n
        marginals = np.zeros(n, dtype=np.float32)
        t = np.zeros(n, dtype=np.int32)
        
        # precompute grand and empty coalition values
        self.func_calls += 2
        self.v_n = self.game.value(np.arange(n))
        self.v_0 = self.game.value(np.array([]))
        
        while self.func_calls+2 <= self.T:
            length = np.random.randint(n+1)
            for player in range(n):
                if self.func_calls+2 > self.T:
                    break
                S = self.sample(length=length)
                marginals[player] += self.marginal(S, player)
                t[player] += 1
            self.update_phi(marginals / t)
        self.func_calls = self.T
        self.save_steps(step_interval)
        
class Strat_CMCS(CMCS):
    def calc_phi(self, marginals, t):
        count = np.sum(t!=0, axis=1)
        t[t==0] = 1
        return np.sum(marginals / t, axis=1)/count
    def get_top_k(self, k: int, step_interval: int = 100):
        # initialization
        self.step_interval = step_interval
        n = self.game.n
        marginals = np.zeros((n, n+1), dtype=np.float32)
        t = np.zeros((n, n+1), dtype=np.int32)
        
        # precompute grand and empty coalition values
        self.func_calls += 2
        self.v_n = self.game.value(np.arange(n))
        self.v_0 = self.game.value(np.array([]))
        
        
        while self.func_calls+2*(n+1) <= self.T:
            for length in range(n+1):
                S = self.sample(length)
                v_S = self.value(S)
                for player in range(n):
                    if self.func_calls == self.T:
                        self.update_phi(self.calc_phi(marginals, t))
                        return
                    marginals[player, length] += self.marginal(S, player, v_S)    
                    t[player, length] += 1
                self.update_phi(self.calc_phi(marginals, t)) 
            
        self.func_calls = self.T
        self.save_steps(step_interval)

class Variance_CMCS(Algorithm):
    def value(self, S: list):
        l = len(S)
        if l == 0:
            v = self.v_0
        elif l == self.game.n:
            v = self.v_n
        else:
            assert self.func_calls < self.T
            self.func_calls += 1
            v = self.game.value(S)
        return v
    def sample(self):
        length = np.random.randint(self.game.n+1)
        S = np.arange(self.game.n)
        np.random.shuffle(S)
        return S[:length], S[length:]
    def get_top_k(self, k: int, step_interval: int = 100):
        self.step_interval = step_interval
        n = self.game.n
        self.phi = np.zeros(n, dtype=np.float32)
        t = np.zeros(n, dtype=np.int32)
        self.func_calls += 2
        self.v_n = self.game.value(np.arange(n))
        self.v_0 = self.game.value(np.array([]))
        
        sum_marginal_prods = np.zeros((n,n), dtype=np.float32)
        count_marginal_prods = np.zeros((n,n), dtype=np.float32)
        diff_variance = np.zeros((n,n), dtype=np.float32)
        
        sum_marginal_squares = np.zeros(n, dtype=np.float32)
        sum_marginals = np.zeros(n, dtype=np.float32)
        sample_variance = np.zeros(n, dtype=np.float32)
        
        selected_players = np.array([])
        while self.func_calls+2 <= self.T:
            
            if np.any(t<2) or selected_players.shape[0] == 0:
                selected_players = np.arange(n)
            else:
                sorted_players = np.argsort(-self.phi)
                # border = (self.phi[sorted_players[k-1]] + self.phi[sorted_players[k]])/2
                partners = np.zeros(n, dtype=np.int32)
                partners[sorted_players[:k]] = sorted_players[k]
                partners[sorted_players[k:]] = sorted_players[k-1]
                var = diff_variance[np.arange(n, dtype=np.int32), partners]
                distance = np.abs(self.phi - self.phi[partners])
                # print(var / distance)
                # var[var == 0] = np.min(var[var != 0])
                certainty = distance / np.sqrt(var) * t
                min_certainty, max_certainty = np.min(certainty), np.max(certainty)
                weights = (max_certainty - certainty) / (max_certainty - min_certainty)
                selected_players = np.array((np.random.rand(n) < weights).nonzero())[0]
            
            S, notS = self.sample()
            v_S = self.value(S)
            marginals = np.zeros(n)
            for player in selected_players:
                if self.func_calls == self.T:
                    self.phi = sum_marginals / t
                    self.save_steps(step_interval)
                    return
                if player in S:
                    marginal = v_S - self.value(S[S != player])
                else:
                    marginal = self.value(np.concatenate((S, [player]))) - v_S
                marginals[player] = marginal
                t[player] += 1
                count_marginal_prods[player, selected_players] += 1
                sum_marginal_squares[player] += marginal**2
                sum_marginals[player] += marginal
            sum_marginal_prods += marginals[np.arange(n, dtype=int), np.newaxis] * marginals
            
            self.phi = sum_marginals / t
            self.save_steps(step_interval)
            if np.all(t > 1):
                sample_variance = (sum_marginal_squares - (sum_marginals**2)/t)/(t-1)
                diff_variance = (sample_variance[np.arange(n, dtype=int), np.newaxis] + sample_variance) + 2 * (self.phi[np.arange(n, dtype=int), np.newaxis] * self.phi) - 2 * sum_marginal_prods / count_marginal_prods
                diff_variance /= count_marginal_prods
                # print(diff_variance[:5,:5])
        self.func_calls = self.T
        self.save_steps(step_interval)

class Variance_CMCS_Cheat(Algorithm):
    def __init__(self, diff_variance=None):
        self.diff_variance = diff_variance
    def value(self, S: list):
        l = len(S)
        if l == 0:
            v = self.v_0
        elif l == self.game.n:
            v = self.v_n
        else:
            assert self.func_calls < self.T
            self.func_calls += 1
            v = self.game.value(S)
        return v
    def sample(self):
        length = np.random.randint(self.game.n+1)
        S = np.arange(self.game.n)
        np.random.shuffle(S)
        return S[:length], S[length:]
    def get_top_k(self, k: int, step_interval: int = 100):
        # print("test")
        if self.diff_variance is None:
            diff_variance = util.calc_variance(self.game.n, self.game.values[np.newaxis, :], self.game.phi[np.newaxis, :])[0]
        else:
            diff_variance = self.diff_variance
        self.step_interval = step_interval
        n = self.game.n
        self.phi = np.zeros(n, dtype=np.float32)
        t = np.zeros(n, dtype=np.int32)
        self.func_calls += 2
        self.v_n = self.game.value(np.arange(n))
        self.v_0 = self.game.value(np.array([]))
        sum_marginals = np.zeros(n, dtype=np.float32)
        
        selected_players = np.array([])
        while self.func_calls+2 <= self.T:
            
            if np.any(t<1) or selected_players.shape[0] == 0:
                selected_players = np.arange(n)
            else:
                sorted_players = np.argsort(-self.phi)
                partners = np.zeros(n, dtype=np.int32)
                partners[sorted_players[:k]] = sorted_players[k]
                partners[sorted_players[k:]] = sorted_players[k-1]
                # var = diff_variance[np.arange(n, dtype=np.int32), partners]
                distance = np.abs(self.phi - self.phi[partners])
                certainty = distance / np.sqrt(diff_variance) * t
                min_certainty, max_certainty = np.min(certainty), np.max(certainty)
                weights = (max_certainty - certainty) / (max_certainty - min_certainty)
                selected_players = np.array((np.random.rand(n) < weights).nonzero())[0]
            
            S, notS = self.sample()
            v_S = self.value(S)
            for player in selected_players:
                if self.func_calls == self.T:
                    self.phi = sum_marginals / t
                    self.save_steps(step_interval)
                    return
                if player in S:
                    marginal = v_S - self.value(S[S != player])
                else:
                    marginal = self.value(np.concatenate((S, [player]))) - v_S
                t[player] += 1
                sum_marginals[player] += marginal
            
            self.phi = sum_marginals / t
            self.save_steps(step_interval)
        self.func_calls = self.T
        self.save_steps(step_interval)

def binom(n, k):
    return math.factorial(n)/(math.factorial(k)*math.factorial(n-k))
class SIR_CMCS(Algorithm):
    def value(self, S: list):
        l = len(S)
        if l == 0:
            v = self.v_0
        elif l == self.game.n:
            v = self.v_n
        else:
            assert self.func_calls < self.T
            self.func_calls += 1
            v = self.game.value(S)
        return v
    def sample(self):
        length = np.random.randint(self.game.n+1)
        S = np.arange(self.game.n)
        np.random.shuffle(S)
        return S[:length], S[length:]
    def get_top_k(self, k: int, step_interval: int = 100):
        self.step_interval = step_interval
        n = self.game.n
        self.phi = np.zeros(n)
        t = 0
        self.func_calls += 2
        self.v_n = self.game.value(np.arange(n))
        self.v_0 = self.game.value(np.array([]))
        
        player_weights = np.zeros(n)
        while self.func_calls+n+1 <= self.T/2:
            S, notS = self.sample()
            v_S = self.value(S)
            for player in S:
                marginal = v_S - self.value(S[S != player])
                self.phi[player] = (t*self.phi[player]+marginal)/(t+1)
                self.save_steps(step_interval)
                player_weights[S] += abs(marginal)
            for player in notS:
                marginal = self.value(np.concatenate((S, [player]))) - v_S
                self.phi[player] = (t*self.phi[player]+marginal)/(t+1)
                self.save_steps(step_interval)
                player_weights[notS] += abs(marginal)
            t += 1
        # for p in range(n):
        #     assert player_weights[p] > 0
        pre_samples = 100
        total_weights = player_weights.mean()
        for player in range(n):
            for length in range(1, n+1):
                total_weights += binom(n-1, length-1) * player_weights[player] / length
        # player_weights /= total_weights
        coalition_weights = np.array([1/((n+1)*binom(n, l)) for l in range(n+1)])
        
        self.phi = np.zeros(n)
        t = 0
        while self.func_calls+n+1 <= self.T:
            total_weight_pre = 0
            for i in range(pre_samples):
                S_new, notS_new = self.sample()
                length_new = S_new.shape[0]
                if length_new == 0:
                    weight = player_weights.mean()
                else:
                    weight = player_weights[S_new].mean()
                # weight /= coalition_weights[length]
                total_weight_pre += weight
                if np.random.rand() < weight/total_weight_pre:
                    S, notS = S_new, notS_new
                    length = length_new
            if length == 0:
                pdf = player_weights.mean() / total_weights
            else:
                pdf = player_weights[S].mean() / total_weights
            v_S = self.value(S)
            for player in S:
                marginal = v_S - self.value(S[S != player])
                marginal *= 1 / pdf
                self.phi[player] = (t*self.phi[player]+marginal)/(t+1)
                self.save_steps(step_interval)
            for player in notS:
                marginal = self.value(np.concatenate((S, [player]))) - v_S
                marginal *= 1 / pdf
                self.phi[player] = (t*self.phi[player]+marginal)/(t+1)
                self.save_steps(step_interval)
            t += 1
            
        self.func_calls = self.T
        self.save_steps(step_interval)
        
class Adaptive_CMCS(Algorithm):
    def value(self, S: list):
        l = len(S)
        if l == 0:
            v = self.v_0
        elif l == self.game.n:
            v = self.v_n
        else:
            assert self.func_calls < self.T
            self.func_calls += 1
            v = self.game.value(S)
        return v
    def sample(self, length=None):
        if length is None:
            length = np.random.randint(self.game.n+1)
        S = np.arange(self.game.n)
        np.random.shuffle(S)
        return S[:length], S[length:]
    def get_top_k(self, k: int, step_interval: int = 100):
        self.step_interval = step_interval
        n = self.game.n
        self.phi = np.zeros(n)
        t = 0
        self.func_calls += 2
        self.v_n = self.game.value(np.arange(n))
        self.v_0 = self.game.value(np.array([]))
        weights_include = np.zeros((2, n), dtype=np.float32)
        weights_exclude = np.zeros((2, n), dtype=np.float32)
        counts_include = np.zeros((2, n), dtype=np.int32)
        counts_exclude = np.zeros((2, n), dtype=np.int32)
        sides = np.zeros(2)
        coalition_weights = np.array([1/((n+1)*binom(n, l)) for l in range(n+1)])
        while self.func_calls+n+1 <= self.T/2 or (np.any(sides == 0) and self.func_calls+n+1 <= self.T):
            S, notS = self.sample()
            length = S.shape[0]
            side = int(length > n/2)
            sides[side] += 1
            v_S = self.value(S)
            absolute_marginals = 0
            for player in S:
                marginal = v_S - self.value(S[S != player])
                self.phi[player] = (t*self.phi[player]+marginal)/(t+1)
                absolute_marginals += abs(marginal)
                self.save_steps(step_interval)
            for player in notS:
                marginal = self.value(np.concatenate((S, [player]))) - v_S
                self.phi[player] = (t*self.phi[player]+marginal)/(t+1)
                absolute_marginals += abs(marginal)
                self.save_steps(step_interval)
            t += 1
            weights_include[side, S] += absolute_marginals 
            counts_include[side, S] += 1
            weights_exclude[side, notS] += absolute_marginals
            counts_exclude[side, notS] += 1
            # print("A", self.func_calls)
        assert not np.any(sides == 0) or self.func_calls+n+1 > self.T
        counts_exclude[counts_exclude == 0] = 1
        counts_include[counts_include == 0] = 1
        weights_include /= counts_include
        weights_exclude /= counts_exclude
        weights_include[weights_include + weights_exclude == 0] = 1
        weights_exclude[weights_include + weights_exclude == 0] = 1
        weights = weights_include / (weights_include + weights_exclude)
        
        self.phi = np.zeros(n)
        t = 0
        # for l in range(n+1):
        #     print(l, weights[l])
        while self.func_calls+n+1 <= self.T:
            p_side = [1 - (math.ceil(n/2) / (n+1)), math.ceil(n/2) / (n+1)] #p_side[0] = P(length <= n/2)
            side = int(np.random.rand() < p_side[1])
            
            length = (1-side)*n # if side=1 length = 0 if side = 0 length = n => either way length is not on the right side
            while (side==0 and length > n/2) or (side==1 and length <= n/2):
                # print(t, "meh")
                random = np.random.rand(n)
                S = np.array((random < weights[side]).nonzero())[0]
                notS = np.array((random >= weights[side]).nonzero())[0]
                density = p_side[side] * np.prod(weights[side, S]) * np.prod(1-weights[side, notS])
                length = S.shape[0]
            
            v_S = self.value(S)
            for player in S:
                marginal = v_S - self.value(S[S != player])
                marginal *= coalition_weights[length] / density
                self.phi[player] = (t*self.phi[player]+marginal)/(t+1)
                self.save_steps(step_interval)
            for player in notS:
                marginal = self.value(np.concatenate((S, [player]))) - v_S
                marginal *= coalition_weights[length] / density
                self.phi[player] = (t*self.phi[player]+marginal)/(t+1)
                self.save_steps(step_interval)
            t += 1
            # print("B", self.func_calls)
            
        
        self.func_calls = self.T
        self.save_steps(step_interval)
        
class Adaptive_CMCS2(Algorithm):
    def value(self, S: list):
        l = len(S)
        if l == 0:
            v = self.v_0
        elif l == self.game.n:
            v = self.v_n
        else:
            assert self.func_calls < self.T
            self.func_calls += 1
            v = self.game.value(S)
        return v
    def sample(self, length=None):
        if length is None:
            length = np.random.randint(self.game.n+1)
        S = np.arange(self.game.n)
        np.random.shuffle(S)
        return S[:length], S[length:]
    def get_top_k(self, k: int, step_interval: int = 100):
        self.step_interval = step_interval
        n = self.game.n
        self.phi = np.zeros(n)
        t = 0
        self.func_calls += 2
        self.v_n = self.game.value(np.arange(n))
        self.v_0 = self.game.value(np.array([]))
        weights_include = np.zeros(n, dtype=np.float32)
        weights_exclude = np.zeros(n, dtype=np.float32)
        counts_include = np.zeros(n, dtype=np.int32)
        counts_exclude = np.zeros(n, dtype=np.int32)
        weights = np.ones(n)/2
        coalition_weights = np.array([1/((n+1)*binom(n, l)) for l in range(n+1)])
        pre_samples = 20
        uncertain_players = np.arange(n)
        while self.func_calls+n+1 <= self.T:
            total_weights = 0
            counter = 0
            while counter < pre_samples:
                counter+=1
                S_new, notS_new = self.sample()
                weight = np.prod(weights[S_new]) * np.prod(1-weights[notS_new]) # coalition_weights[lenght] / coalition_weights[lenght]
                if weight == 0:
                    counter-=1
                    continue
                total_weights += weight
                if np.random.rand() < weight / total_weights:
                    S, notS = S_new, notS_new
            length = S.shape[0]
            density = np.prod(weights[S]) * np.prod(1-weights[notS]) #* coalition_weights[length]
            bias_reduction_factor = total_weights / pre_samples
            v_S = self.value(S)
            absolute_marginals = 0
            for player in S:
                marginal = v_S - self.value(S[S != player])
                marginal *= bias_reduction_factor / density
                self.phi[player] = (t*self.phi[player]+marginal)/(t+1)
                if player in uncertain_players:
                    absolute_marginals += abs(marginal)
                self.save_steps(step_interval)
            for player in notS:
                marginal = self.value(np.concatenate((S, [player]))) - v_S
                marginal *= bias_reduction_factor / density
                self.phi[player] = (t*self.phi[player]+marginal)/(t+1)
                if player in uncertain_players:
                    absolute_marginals += abs(marginal)
                self.save_steps(step_interval)
            weights_include[S] += absolute_marginals
            counts_include[S] += 1
            weights_exclude[notS] += absolute_marginals
            counts_exclude[notS] += 1
            t += 1
            
            if t % 20 == 0:
                self.phi = np.zeros(n)
                # prev_t += t
                t = 0
                counts_include[counts_include == 0] = 1
                counts_exclude[counts_exclude == 0] = 1
                weights_include /= counts_include
                weights_exclude /= counts_exclude
                weights_include[weights_include + weights_exclude == 0] = 1
                weights_exclude[weights_include + weights_exclude == 0] = 1
                weights = weights_include / (weights_include + weights_exclude)
                # print(weights)
                weights_include = np.zeros(n, dtype=np.float32)
                weights_exclude = np.zeros(n, dtype=np.float32)
                counts_include = np.zeros(n, dtype=np.int32)
                counts_exclude = np.zeros(n, dtype=np.int32)
                sorted = np.argsort(-self.phi)
                border = (self.phi[sorted[k-1]] + self.phi[sorted[k]])/2
                distances = (self.phi - border)**2
                num_uncertain_players = 4
                uncertain_players = np.argsort(distances)[:num_uncertain_players]
            
        
        self.func_calls = self.T
        self.save_steps(step_interval)

class Adaptive_Stratified_CMCS(Algorithm):
    def value(self, S: list):
        l = len(S)
        if l == 0:
            v = self.v_0
        elif l == self.game.n:
            v = self.v_n
        else:
            assert self.func_calls < self.T
            self.func_calls += 1
            v = self.game.value(S)
        return v
    def sample(self, length=None, weights=None):
        if weights is not None:
            included_players = np.array((weights==1).nonzero())[0]
            excluded_players = np.array((weights==0).nonzero())[0]
            sub_length = length - included_players.shape[0]
            _S = np.setdiff1d(np.arange(self.game.n), np.concatenate((included_players, excluded_players)))
            np.random.shuffle(_S)
            S, notS = np.concatenate(( _S[:sub_length], included_players)), np.concatenate((_S[sub_length:], excluded_players))
            assert S.shape[0] == length
            return S, notS
        if length is None:
            length = np.random.randint(self.game.n+1)
        S = np.arange(self.game.n)
        np.random.shuffle(S)
        return S[:length], S[length:]
    def get_top_k(self, k: int, step_interval: int = 100):
        self.step_interval = step_interval
        n = self.game.n
        self.phi = np.zeros(n)
        t = 0
        self.func_calls += 2
        self.v_n = self.game.value(np.arange(n))
        self.v_0 = self.game.value(np.array([]))
        weights_include = np.zeros(n, dtype=np.float32)
        weights_exclude = np.zeros(n, dtype=np.float32)
        counts_include = np.zeros(n, dtype=np.int32)
        counts_exclude = np.zeros(n, dtype=np.int32)
        phi_outer = np.zeros(n, dtype=np.float32)
        counts_outer = np.zeros(n, dtype=np.int32)
        while self.func_calls+n+1 <= self.T/4:
            S, notS = self.sample()
            length = S.shape[0]
            len_in_outer = length <= n/4 or length > n*3/4
            v_S = self.value(S)
            absolute_marginals = 0
            for player in S:
                marginal = v_S - self.value(S[S != player])
                self.phi[player] = (t*self.phi[player]+marginal)/(t+1)
                if len_in_outer:
                    phi_outer[player] = (counts_outer[player] * phi_outer[player] + marginal) / (counts_outer[player] + 1)
                    counts_outer[player] += 1
                absolute_marginals += abs(marginal)
                self.save_steps(step_interval)
            for player in notS:
                marginal = self.value(np.concatenate((S, [player]))) - v_S
                self.phi[player] = (t*self.phi[player]+marginal)/(t+1)
                if len_in_outer:
                    phi_outer[player] = (counts_outer[player] * phi_outer[player] + marginal) / (counts_outer[player] + 1)
                    counts_outer[player] += 1
                absolute_marginals += abs(marginal)
                self.save_steps(step_interval)
            t += 1
            if not len_in_outer: 
                weights_include[S] += absolute_marginals
                counts_include[S] += 1
                weights_exclude[notS] += absolute_marginals
                counts_exclude[notS] += 1
        
        self.phi = np.zeros(n)
        t = 0
        coalition_weights = np.array([1/((n+1)*binom(n, l)) for l in range(n+1)])
        # for l in range(n+1):
        #     print(l, weights[l])
        p_inside = (math.ceil(n*3/4)-math.ceil(n/4)) / (n+1)
        phi_in = np.zeros(n)
        while self.func_calls+n+1 <= self.T:
            if t % 100 == 0:
                weights_include /= counts_include
                weights_exclude /= counts_exclude
                weights_include[counts_include == 0] = 0
                weights_exclude[counts_exclude == 0] = 0
                weights = weights_include / (weights_include + weights_exclude)
                weights[(weights_include + weights_exclude) == 0] = 0.5
                weights_include = np.zeros(n, dtype=np.float32)
                weights_exclude = np.zeros(n, dtype=np.float32)
                counts_include = np.zeros(n, dtype=np.int32)
                counts_exclude = np.zeros(n, dtype=np.int32)
                
                sorted = np.argsort(-self.phi)
                border = (self.phi[sorted[k-1]] - self.phi[sorted[k]])/2
                distances = (self.phi - border)**2
                num_uncertain_players = 1
                uncertain_players = np.argsort(distances)[:num_uncertain_players]
                
            length = 0 # if side=1 length = 0 if side = 0 length = n => either way length is not on the right side
            counter = 0
            while length <= n/4 or length > n*3/4:
                counter+=1
                assert counter < 100, weights
                # print(t, "meh")
                random = np.random.rand(n)
                S = np.array((random < weights).nonzero())[0]
                notS = np.array((random >= weights).nonzero())[0]
                density = np.prod(weights[S]) * np.prod(1-weights[notS])
                length = S.shape[0]
            v_S = self.value(S)
            absolute_marginals = 0
            for player in S:
                marginal = v_S - self.value(S[S != player])
                marginal *= coalition_weights[length] / density
                phi_in[player] = (t*phi_in[player]+marginal)/(t+1)
                if player in uncertain_players:
                    absolute_marginals += abs(marginal)
            for player in notS:
                marginal = self.value(np.concatenate((S, [player]))) - v_S
                marginal *= coalition_weights[length] / density
                phi_in[player] = (t*phi_in[player]+marginal)/(t+1)
                if player in uncertain_players:
                    absolute_marginals += abs(marginal)
            weights_include[S] += absolute_marginals / density
            counts_include[S] += 1
            weights_exclude[notS] += absolute_marginals / density
            counts_exclude[notS] += 1
            
            self.phi = p_inside*phi_in + (1-p_inside) * phi_outer
            self.save_steps(step_interval)
            t += 1
            # print("B", self.func_calls)
            
        
        self.func_calls = self.T
        self.save_steps(step_interval)