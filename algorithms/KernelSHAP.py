from .algorithm import Algorithm
import numpy as np
import shap

class KernelSHAP(Algorithm):
    def initialize(self, game, budget: int):
        self.game = game
        self.budget = budget
        self.func_calls = 0
        def model(X):
            result = np.zeros(X.shape[0])
            self.func_calls += X.shape[0]
            for i, sample in enumerate(X):
                result[i] = self.game.value(np.where(sample)[0])
            return result
        self.n = self.game.n
        data = np.zeros((1, self.n))
        self.explainer = shap.KernelExplainer(model=model, data=data, feature_names=np.arange(self.n))
    def get_top_k(self, k: int, step_interval: int = 100):
        n = self.n
        self.values = []
        for budget in range(step_interval, self.budget+1, step_interval):
            self.func_calls = 0
            self.values += [self.explainer.shap_values(np.ones(n), nsamples=budget-1, l1_reg=f"num_features({self.n})")]
            assert self.func_calls == budget, (self.func_calls, budget)
        # print(self.values)
        