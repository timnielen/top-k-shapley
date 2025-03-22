from .algorithm import Algorithm
import numpy as np
import shap

class KernelSHAP(Algorithm):
    '''
    wrapper for the sota implementation in the shap package
    '''
    def initialize(self, game, budget: int, step_interval: int = 100):
        self.step_interval = step_interval
        self.game = game
        self.budget = budget
        self.func_calls = 0
        # the kernel explainer expects a model that returns a batch of values of coalitions
        def model(X):
            result = np.zeros(X.shape[0])
            self.func_calls += X.shape[0]
            for i, sample in enumerate(X):
                result[i] = self.game.value(np.where(sample)[0])
            return result
        self.n = self.game.n
        data = np.zeros((1, self.n)) # this is used by the explainer to determine how to fill features (players) that are absent in the coalition as it expects an input of fixed shape to the model
        self.explainer = shap.KernelExplainer(model=model, data=data, feature_names=np.arange(self.n))
    def get_top_k(self, k: int):
        assert self.budget != -1, "The algorithm doesn't have an early stopping condition!"
        n = self.n
        self.values = []
        for budget in range(self.step_interval, self.budget+1, self.step_interval):
            self.func_calls = 0
            '''
            - the explainer uses nsamples+1 value function calls so we use nsamples=budget-1
            - we want all players' shapley values therefore we use X=np.ones(n) and l1_reg=f"num_features({self.n})"
            '''
            self.values += [self.explainer.shap_values(np.ones(n), nsamples=budget-1, l1_reg=f"num_features({self.n})")] 
            assert self.func_calls == budget, (self.func_calls, budget)
            
        