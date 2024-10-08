import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random
import logging
import matplotlib.pyplot as plt
from collections import deque
import json
import talib  # 需要安装 TA-Lib

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingSystem:
    def __init__(self, csv_file, initial_balance=100000, start_date='2024-09-01'):
        self.csv_file = csv_file
        self.balance = initial_balance
        self.position = 0  # 0表示无仓位，1表示多头，-1表示空头
        self.trades = []
        self.open_trades = []  # 用于跟踪开仓信息
        self.data = None
        self.current_index = 0
        self.ma_short = deque(maxlen=10)
        self.ma_long = deque(maxlen=30)
        self.last_trade_index = -1
        self.cooling_period = 40  # 增加冷却期
        self.start_date = pd.to_datetime(start_date)

    def load_data(self):
        try:
            self.data = pd.read_csv(self.csv_file)
            self.data['datetime'] = pd.to_datetime(self.data['datetime'])
            self.data = self.data.sort_values('datetime')
            
            # 只保留开始日期之后的数据
            self.data = self.data[self.data['datetime'] >= self.start_date]
            
            logger.info(f"数据加载成功，共 {len(self.data)} 条记录，从 {self.data['datetime'].min()} 到 {self.data['datetime'].max()}")
        except Exception as e:
            logger.error(f"加载数据时出错: {str(e)}")
            raise

    def preprocess_data(self):
        try:
            # 计算收益率
            self.data['Returns'] = self.data['close'].pct_change()
            
            # 计算波动率
            self.data['Volatility'] = self.data['Returns'].rolling(window=20).std() * np.sqrt(252)
            
            # 计算移动平均线
            self.data['MA10'] = self.data['close'].rolling(window=10).mean()
            self.data['MA30'] = self.data['close'].rolling(window=30).mean()
            
            # 计算RSI
            delta = self.data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            self.data['RSI'] = 100 - (100 / (1 + rs))
            
            # 添加 MACD 指标
            self.data['MACD'], self.data['Signal'], _ = talib.MACD(self.data['close'])
            
            # 删除包含 NaN 的行
            self.data = self.data.dropna()
            
            logger.info("数据预处理完成")
        except Exception as e:
            logger.error(f"预处理数据时出错: {str(e)}")
            raise

    def update_moving_averages(self, price):
        self.ma_short.append(price)
        self.ma_long.append(price)

    def get_current_ma(self):
        return np.mean(self.ma_short), np.mean(self.ma_long)

    def generate_signal(self, row):
        ma_short, ma_long = self.get_current_ma()
        rsi = row['RSI']
        price = row['close']
        macd = row['MACD']
        signal = row['Signal']
        
        # 使用更严格的条件生成信号
        if ma_short > ma_long and rsi < 40 and price > row['low'] and macd > signal:
            return 1  # 买入信号
        elif ma_short < ma_long and rsi > 60 and price < row['high'] and macd < signal:
            return -1  # 卖出信号
        else:
            return 0  # 持有信号

    def calculate_position_size(self, price):
        account_value = self.calculate_portfolio_value()
        risk_per_trade = 0.01 * account_value  # 风险1%的账户价值
        volatility = self.data['Volatility'].iloc[self.current_index]
        if np.isnan(volatility) or np.isnan(price):
            return 0
        contract_size = max(1, int(risk_per_trade / (price * volatility)))
        return min(contract_size, 5)  # 最大持仓限制为5手

    def calculate_portfolio_value(self):
        price = self.data['close'].iloc[self.current_index]
        unrealized_pnl = 0
        if self.open_trades:
            unrealized_pnl = self.position * (price - self.open_trades[0]['price'])
        return self.balance + unrealized_pnl

    def execute_trade(self, signal, price):
        if signal == 0 or np.isnan(price):
            return

        # 检查是否在冷却期内
        if self.current_index - self.last_trade_index <= self.cooling_period:
            return

        timestamp = self.data['datetime'].iloc[self.current_index]

        if signal == 1 and self.position <= 0:  # 买入信号且当前无多头仓位
            contract_size = self.calculate_position_size(price)
            if self.position == -1:  # 如果有空头仓位，先平仓
                profit = self.open_trades[0]['price'] - price
                self.balance += profit
                self.trades.append(('平空', 1, self.open_trades[0]['price'], self.open_trades[0]['time'], price, timestamp, profit))
                self.open_trades.pop(0)
                self.position = 0

            # 开多仓
            self.position = 1
            self.open_trades.append({'price': price, 'size': contract_size, 'time': timestamp})
            self.trades.append(('开多', contract_size, price, timestamp, None, None, None))
            logger.info(f"开多 {contract_size} 手，价格：{price:.2f}")

        elif signal == -1 and self.position >= 0:  # 卖出信号且当前无空头仓位
            contract_size = self.calculate_position_size(price)
            if self.position == 1:  # 如果有多头仓位，先平仓
                profit = price - self.open_trades[0]['price']
                self.balance += profit
                self.trades.append(('平多', 1, self.open_trades[0]['price'], self.open_trades[0]['time'], price, timestamp, profit))
                self.open_trades.pop(0)
                self.position = 0

            # 开空仓
            self.position = -1
            self.open_trades.append({'price': price, 'size': -contract_size, 'time': timestamp})
            self.trades.append(('开空', contract_size, price, timestamp, None, None, None))
            logger.info(f"开空 {contract_size} 手，价格：{price:.2f}")

        # 更新最后交易时间
        self.last_trade_index = self.current_index

    def check_stop_loss_take_profit(self):
        if not self.open_trades:
            return

        current_price = self.data['close'].iloc[self.current_index]
        open_trade = self.open_trades[0]
        open_price = open_trade['price']
        timestamp = self.data['datetime'].iloc[self.current_index]

        stop_loss_pct = 0.015  # 1.5%止损
        take_profit_pct = 0.025  # 2.5%止盈

        if self.position == 1:  # 多头
            if current_price <= open_price * (1 - stop_loss_pct) or current_price >= open_price * (1 + take_profit_pct):
                profit = current_price - open_price
                self.balance += profit
                self.trades.append(('平多', 1, open_price, open_trade['time'], current_price, timestamp, profit))
                self.open_trades.pop(0)
                self.position = 0
                logger.info(f"触发{'止损' if profit < 0 else '止盈'}，平多 1 手，价格：{current_price:.2f}, 盈亏：{profit:.2f}")

        elif self.position == -1:  # 空头
            if current_price >= open_price * (1 + stop_loss_pct) or current_price <= open_price * (1 - take_profit_pct):
                profit = open_price - current_price
                self.balance += profit
                self.trades.append(('平空', 1, open_price, open_trade['time'], current_price, timestamp, profit))
                self.open_trades.pop(0)
                self.position = 0
                logger.info(f"触发{'止损' if profit < 0 else '止盈'}，平空 1 手，价格：{current_price:.2f}, 盈亏：{profit:.2f}")

    def run_simulation(self):
        logger.info("开始交易模拟")
        portfolio_values = []

        for i in range(len(self.data)):
            self.current_index = i
            row = self.data.iloc[i]
            price = row['close']
            
            if not np.isnan(price):
                self.update_moving_averages(price)
                self.check_stop_loss_take_profit()  # 添加这行
                signal = self.generate_signal(row)
                self.execute_trade(signal, price)
            
            portfolio_value = self.calculate_portfolio_value()
            portfolio_values.append(portfolio_value)
            
            if i % 100 == 0:
                logger.info(f"模拟进度: {i}/{len(self.data)}")

        self.plot_portfolio_value(portfolio_values)
        self.print_summary(portfolio_values)

    def plot_portfolio_value(self, portfolio_values):
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['datetime'], portfolio_values)
        plt.title('投资组合价值随时间变化')
        plt.xlabel('时间')
        plt.ylabel('投资组合价值')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('portfolio_value.png')
        logger.info("投资组合价值图表已保存为 portfolio_value.png")

    def print_summary(self, portfolio_values):
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value * 100
        
        logger.info("=== 交易摘要 ===")
        logger.info(f"初始资金: ${initial_value:.2f}")
        logger.info(f"最终资金: ${final_value:.2f}")
        logger.info(f"总回报率: {total_return:.2f}%")
        logger.info(f"总交易次数: {len(self.trades)}")
        
        logger.info("\n=== 详细交易记录 ===")
        
        trade_records = []
        open_trades = {}
        total_profit = 0
        
        for trade in self.trades:
            if trade[0] in ['开多', '开空']:
                open_trades[trade[3]] = trade  # 使用开仓时间作为键
            else:  # 平仓交易
                open_trade = open_trades.pop(trade[3], None)
                if open_trade:
                    trade_type = '多' if open_trade[0] == '开多' else '空'
                    size = trade[1]
                    open_price = open_trade[2]
                    open_time = open_trade[3]
                    close_price = trade[4]
                    close_time = trade[5]
                    profit = trade[6]
                    total_profit += profit
                    
                    trade_record = {
                        "交易类型": trade_type,
                        "手数": size,
                        "开仓价格": round(open_price, 2),
                        "开仓时间": open_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "平仓价格": round(close_price, 2),
                        "平仓时间": close_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "盈利": round(profit, 2)
                    }
                    trade_records.append(trade_record)
        
        # 输出未平仓的交易
        for open_trade in open_trades.values():
            trade_type = '多' if open_trade[0] == '开多' else '空'
            size = open_trade[1]
            open_price = open_trade[2]
            open_time = open_trade[3]
            
            trade_record = {
                "交易类型": trade_type,
                "手数": size,
                "开仓价格": round(open_price, 2),
                "开仓时间": open_time.strftime("%Y-%m-%d %H:%M:%S"),
                "平仓价格": "未平仓",
                "平仓时间": "未平仓",
                "盈利": "未实现"
            }
            trade_records.append(trade_record)
        
        # 将交易记录转换为JSON格式并输出
        json_output = json.dumps(trade_records, ensure_ascii=False, indent=2)
        logger.info(f"交易记录 JSON:\n{json_output}")
        
        logger.info(f"\n最终持仓: {self.position} 手")
        logger.info(f"现金余额: ${self.balance:.2f}")
        logger.info(f"总盈利: ${total_profit:.2f}")

        # 计算夏普比率
        returns = pd.Series(portfolio_values).pct_change().dropna()
        sharpe_ratio = PerformanceAnalysis.calculate_sharpe_ratio(returns)

        # 计算最大回撤
        max_drawdown = PerformanceAnalysis.calculate_max_drawdown(portfolio_values)

        logger.info(f"夏普比率: {sharpe_ratio:.2f}")
        logger.info(f"最大回撤: {max_drawdown:.2%}")

        # 计算盈利交易和亏损交易
        profit_trades = [t for t in self.trades if t[6] is not None and t[6] > 0]
        loss_trades = [t for t in self.trades if t[6] is not None and t[6] <= 0]

        logger.info(f"盈利交易次数: {len(profit_trades)}")
        logger.info(f"亏损交易次数: {len(loss_trades)}")
        
        if profit_trades:
            avg_profit = sum(t[6] for t in profit_trades) / len(profit_trades)
            logger.info(f"平均盈利: ${avg_profit:.2f}")
        
        if loss_trades:
            avg_loss = sum(t[6] for t in loss_trades) / len(loss_trades)
            logger.info(f"平均亏损: ${avg_loss:.2f}")

    def run_backtest(self):
        self.load_data()
        self.preprocess_data()
        self.run_simulation()

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
        excess_returns = returns - risk_free_rate / 252  # 假设风险无关收益率为年化2%
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

class PerformanceAnalysis:
    @staticmethod
    def calculate_drawdowns(portfolio_values):
        hwm = np.maximum.accumulate(portfolio_values)
        drawdowns = (hwm - portfolio_values) / hwm
        return drawdowns

    @staticmethod
    def calculate_max_drawdown(portfolio_values):
        drawdowns = PerformanceAnalysis.calculate_drawdowns(portfolio_values)
        return drawdowns.max()

    @staticmethod
    def calculate_cagr(initial_value, final_value, years):
        return (final_value / initial_value) ** (1 / years) - 1

class MarketDataFeed:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.csv_file)
        self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'])
        self.data = self.data.sort_values('Timestamp')

    def get_latest_price(self, symbol):
        latest_data = self.data[self.data['Symbol'] == symbol].iloc[-1]
        return latest_data['Close']

    def get_historical_data(self, symbol, start_date, end_date):
        mask = (self.data['Symbol'] == symbol) & (self.data['Timestamp'] >= start_date) & (self.data['Timestamp'] <= end_date)
        return self.data.loc[mask]

class OrderManagement:
    def __init__(self):
        self.orders = []

    def place_order(self, symbol, order_type, quantity, price):
        order = {
            'symbol': symbol,
            'type': order_type,
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.now(),
            'status': 'PENDING'
        }
        self.orders.append(order)
        return len(self.orders) - 1  # 返回订单ID

    def cancel_order(self, order_id):
        if 0 <= order_id < len(self.orders):
            if self.orders[order_id]['status'] == 'PENDING':
                self.orders[order_id]['status'] = 'CANCELLED'
                return True
        return False

    def get_order_status(self, order_id):
        if 0 <= order_id < len(self.orders):
            return self.orders[order_id]['status']
        return None

class PortfolioManager:
    def __init__(self, initial_balance):
        self.cash_balance = initial_balance
        self.positions = {}

    def update_position(self, symbol, quantity, price):
        if symbol not in self.positions:
            self.positions[symbol] = {'quantity': 0, 'average_price': 0}
        
        current_quantity = self.positions[symbol]['quantity']
        current_average_price = self.positions[symbol]['average_price']
        
        new_quantity = current_quantity + quantity
        if new_quantity > 0:
            new_average_price = (current_quantity * current_average_price + quantity * price) / new_quantity
        else:
            new_average_price = 0
        
        self.positions[symbol]['quantity'] = new_quantity
        self.positions[symbol]['average_price'] = new_average_price

    def get_position(self, symbol):
        return self.positions.get(symbol, {'quantity': 0, 'average_price': 0})

    def update_cash_balance(self, amount):
        self.cash_balance += amount

    def get_total_value(self, market_data_feed):
        total_value = self.cash_balance
        for symbol, position in self.positions.items():
            current_price = market_data_feed.get_latest_price(symbol)
            total_value += position['quantity'] * current_price
        return total_value

def main():
    csv_file = "e:/data/29#CL8.csv"
    initial_balance = 100000
    start_date = '2024-09-01'
    
    trading_system = TradingSystem(csv_file, initial_balance, start_date)
    trading_system.run_backtest()

if __name__ == "__main__":
    main()