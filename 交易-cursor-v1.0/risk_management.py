import numpy as np

class RiskManagement:
    @staticmethod
    def calculate_var(returns, confidence_level=0.95):
        return np.percentile(returns, 100 * (1 - confidence_level))

    @staticmethod
    def calculate_expected_shortfall(returns, confidence_level=0.95):
        var = RiskManagement.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()

    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
        excess_returns = returns - risk_free_rate / 252
        std_dev = excess_returns.std()
        epsilon = 1e-10
        if std_dev <= epsilon:
            return 0
        return np.sqrt(252) * excess_returns.mean() / std_dev
