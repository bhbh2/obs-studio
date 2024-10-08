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
import os
import datetime  # 确保在文件顶部导入了 datetime
import pyautogui
import keyboard
import threading
import queue
import requests
import json
from datetime import datetime

# 在文件开头添加这行代码
pyautogui.FAILSAFE = False  # 禁用 FAILSAFE 模式，但请谨慎使用

# 设置日志级别为 ERROR，这样只有错误信息会被输出
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingSystem:
    def __init__(self, csv_file, initial_balance=100000, start_date='2024-09-01', end_date=None, state_file='trading_state.json'):
        self.csv_file = csv_file
        self.balance = initial_balance
        self.position = 0  # 0表示无仓位，1表示多头，-1表示空头
        self.trades = []
        self.open_trades = []  # 用于跟踪开仓信息
        self.data = None
        self.current_index = 0
        self.ma_short = []
        self.ma_long = []
        self.last_trade_index = -1
        self.cooling_period = 40  # 增加冷却期
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) if end_date else None
        self.total_profit = 0  # 添加这行来跟踪总盈利
        self.point_value = 10  # 一个点的价值为10元
        self.commission = 7.2  # 每次开平仓的总手续费为7.2元
        self.state_file = state_file
        self.paused = False
        self.command_queue = queue.Queue()

    def load_data(self):
        try:
            self.data = pd.read_csv(self.csv_file)
            self.data['datetime'] = pd.to_datetime(self.data['datetime'])
            self.data = self.data.sort_values('datetime')
            
            # 只保留开始日期之后的数据
            self.data = self.data[self.data['datetime'] >= self.start_date]
            
            # 如果设置了结束日期，则只保留结束日期之前的数据
            if self.end_date:
                self.data = self.data[self.data['datetime'] <= self.end_date]
            
            if self.data.empty:
                logger.warning(f"警告：在指定的时间范围内没有数据")
                return

            logger.info(f"数据加载成功，共 {len(self.data)} 条记录，从 {self.data['datetime'].min()} 到 {self.data['datetime'].max()}")
        except Exception as e:
            logger.error(f"加载数据时出错: {str(e)}")
            raise

    def preprocess_data(self):
        try:
            # 计算收益率
            self.data['Returns'] = self.data['close'].pct_change()
            
            # 计算波动率
            self.data['Volatility'] = self.data['Returns'].rolling(window=20, min_periods=1).std() * np.sqrt(252)
            
            # 计算移动平均线
            self.data['MA10'] = self.data['close'].rolling(window=10, min_periods=1).mean()
            self.data['MA30'] = self.data['close'].rolling(window=30, min_periods=1).mean()
            
            # 计算RSI
            delta = self.data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            epsilon = 1e-10  # 添加一个小的值以避免除以零
            rs = np.where(loss != 0, gain / (loss + epsilon), 0)
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
        if len(self.ma_short) > 0 and len(self.ma_long) > 0:
            return self.ma_short[-1], self.ma_long[-1]
        return None, None

    def generate_signal(self, row):
        ma_short, ma_long = self.get_current_ma()
        if ma_short is None or ma_long is None:
            print("没有足够的数据来计算移动平均线")
            return 0

        rsi = row['RSI']
        price = row['close']
        macd = row['MACD']
        signal = row['Signal']
        
        #print(f"生成信号：MA短期 {ma_short:.2f}, MA长期 {ma_long:.2f}, RSI {rsi:.2f}, MACD {macd:.2f}, Signal {signal:.2f}")
        
        if ma_short > ma_long and rsi < 50 and macd > signal:
            #print("生成买入信号")
            return 1
        elif ma_short < ma_long and rsi > 50 and macd < signal:
            #print("生成卖出信号")
            return -1
        else:
            #print("生成持有信号")
            return 0

    def calculate_position_size(self, price):
        try:
            account_value = self.calculate_portfolio_value()
            risk_per_trade = 0.01 * account_value  # 风险1%的账户价值
            volatility = self.data['Volatility'].iloc[self.current_index]
            epsilon = 1e-10  # 添加一个小的值以避免除以零
            if np.isnan(volatility) or np.isnan(price) or volatility <= epsilon or price <= epsilon:
                return 0
            contract_size = max(1, int(risk_per_trade / (price * volatility)))
            return min(contract_size, 5)  # 最大持仓限制为5手
        except Exception as e:
            logger.error(f"计算仓位大小时出错: {str(e)}")
            return 1  # 如果出错，默认返回1手

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
            return None

        current_price = self.data['close'].iloc[-1]
        open_trade = self.open_trades[0]
        open_price = open_trade['price']
        timestamp = self.data['datetime'].iloc[-1]

        stop_loss_pct = 0.015  # 1.5%止损
        take_profit_pct = 0.025  # 2.5%止盈

        if self.position == 1:  # 多头
            if current_price <= open_price * (1 - stop_loss_pct) or current_price >= open_price * (1 + take_profit_pct):
                profit_points = current_price - open_price
                profit = profit_points * self.point_value - self.commission
                self.balance += profit
                self.total_profit += profit
                self.trades.append(('平多', 1, open_price, open_trade['time'], current_price, timestamp, profit))
                self.open_trades.pop(0)
                self.position = 0
                return {
                    "action": "平多(止损/止盈)",
                    "size": 1,
                    "price": current_price,
                    "timestamp": timestamp,
                    "profit": profit,
                    "open_price": open_price,
                    "open_time": open_trade['time']
                }

        elif self.position == -1:  # 空头
            if current_price >= open_price * (1 + stop_loss_pct) or current_price <= open_price * (1 - take_profit_pct):
                profit_points = open_price - current_price
                profit = profit_points * self.point_value - self.commission
                self.balance += profit
                self.total_profit += profit
                self.trades.append(('平空', 1, open_price, open_trade['time'], current_price, timestamp, profit))
                self.open_trades.pop(0)
                self.position = 0
                return {
                    "action": "平空(止损/止盈)",
                    "size": 1,
                    "price": current_price,
                    "timestamp": timestamp,
                    "profit": profit,
                    "open_price": open_price,
                    "open_time": open_trade['time']
                }

        return None

    def keyboard_listener(self):
        def on_press(key):
            if key.name == 'p':
                self.command_queue.put('pause')
            elif key.name == 'o':
                self.command_queue.put('resume')

        keyboard.on_press(on_press)

    def handle_commands(self):
        while True:
            try:
                command = self.command_queue.get_nowait()
                if command == 'pause':
                    self.paused = True
                    print("程序已暂停。按 'o' 继续。")
                elif command == 'resume':
                    self.paused = False
                    print("程序继续运行。")
            except queue.Empty:
                break

    def run_simulation(self):
        logger.info("开始交易模拟")
        portfolio_values = []

        # 启动键盘监听线程
        threading.Thread(target=self.keyboard_listener, daemon=True).start()

        for i in range(len(self.data)):
            self.handle_commands()
            while self.paused:
                time.sleep(0.1)
                self.handle_commands()

            self.current_index = i
            row = self.data.iloc[i]
            price = row['close']
            
            if not np.isnan(price):
                self.update_moving_averages(price)
                self.check_stop_loss_take_profit()
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

    def real_time_trade(self, timestamp, price):
        try:
            self.handle_commands()
            while self.paused:
                time.sleep(0.1)
                self.handle_commands()

            # 确保数据已经加载和预处理
            if self.data is None:
                self.load_data()
                self.preprocess_data()

            # 更新当前索引
            self.current_index = len(self.data)

            # 添加新的数据点
            new_data = pd.DataFrame({
                'datetime': [timestamp],
                'close': [price],
                'high': [price],  # 假设当前价格就是最高价
                'low': [price],   # 假设当前价格就是最低价
                # 其他列可以根据需要添加
            })
            self.data = pd.concat([self.data, new_data], ignore_index=True)

            # 更新指标
            self.update_indicators()

            # 检查止损止盈
            stop_loss_take_profit_action = self.check_stop_loss_take_profit()
            if stop_loss_take_profit_action:
                result = self.format_trade_result(stop_loss_take_profit_action)
                if result['盈利'] != "未实现":
                    self.total_profit += result['盈利']
                return result, self.get_latest_trade_record(), self.total_profit

            # 生成交易信号
            signal = self.generate_signal(self.data.iloc[-1])

            # 执行交易
            trade_action = self.execute_real_time_trade(signal, price, timestamp)

            if trade_action:
                result = self.format_trade_result(trade_action)
                if result['盈利'] != "未实现":
                    self.total_profit += result['盈利']
                return result, self.get_latest_trade_record(), self.total_profit
            
            return None, None, self.total_profit
        except Exception as e:
            logger.error(f"实时交易时出错: {str(e)}")
            return None, None, self.total_profit

    def get_latest_trade_record(self):
        if self.trades:
            latest_trade = self.trades[-1]
            if len(latest_trade) == 7 and latest_trade[4] is not None:  # 平仓交易
                return {
                    "交易类型": latest_trade[0],
                    "手数": latest_trade[1],
                    "开仓价格": round(float(latest_trade[2]), 2),
                    "开仓时间": latest_trade[3].strftime("%Y-%m-%d %H:%M:%S"),
                    "平仓价格": round(float(latest_trade[4]), 2),
                    "平仓时间": latest_trade[5].strftime("%Y-%m-%d %H:%M:%S"),
                    "盈利": round(float(latest_trade[6]), 2)
                }
            else:  # 开仓交易
                return {
                    "交易类型": latest_trade[0],
                    "手数": latest_trade[1],
                    "开仓价格": round(float(latest_trade[2]), 2),
                    "开仓时间": latest_trade[3].strftime("%Y-%m-%d %H:%M:%S"),
                    "平仓价格": "未平仓",
                    "平仓时间": "未平仓",
                    "盈利": "未实现"
                }
        return None

    def update_indicators(self):
        try:
            # 更新移动平均线
            self.data['MA10'] = self.data['close'].rolling(window=10, min_periods=1).mean()
            self.data['MA30'] = self.data['close'].rolling(window=30, min_periods=1).mean()

            # 更新RSI
            delta = self.data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            epsilon = 1e-10
            rs = np.where(loss != 0, gain / (loss + epsilon), 100)
            self.data['RSI'] = 100 - (100 / (1 + rs))

            # 更新MACD
            self.data['MACD'], self.data['Signal'], _ = talib.MACD(self.data['close'], fastperiod=12, slowperiod=26, signalperiod=9)

            # 更新 ma_short 和 ma_long
            self.ma_short = self.data['MA10'].tolist()
            self.ma_long = self.data['MA30'].tolist()

            # 移除打印语句
            # latest_data = self.data.iloc[-1]
            # print(f"Latest indicators: MA10={latest_data['MA10']:.2f}, MA30={latest_data['MA30']:.2f}, RSI={latest_data['RSI']:.2f}, MACD={latest_data['MACD']:.2f}, Signal={latest_data['Signal']:.2f}")

        except Exception as e:
            logger.error(f"更新指标时出错: {str(e)}")
            raise

    def execute_real_time_trade(self, signal, price, timestamp):
        if signal == 0:
            return None

        if self.current_index - self.last_trade_index <= self.cooling_period:
            return None

        contract_size = self.calculate_position_size(price)

        if signal == 1 and self.position <= 0:  # 买入信号
            if self.position == -1:
                if self.open_trades:
                    # 平空
                    open_trade = self.open_trades[0]
                    profit_points = open_trade['price'] - price
                    profit = profit_points * self.point_value - self.commission  # 只在平仓时扣除手续费
                    self.balance += profit
                    self.total_profit += profit
                    trade_action = {
                        "action": "平空",
                        "size": 1,
                        "price": price,
                        "timestamp": timestamp,
                        "profit": profit,
                        "open_price": open_trade['price'],
                        "open_time": open_trade['time']
                    }
                    self.trades.append(('平空', 1, open_trade['price'], open_trade['time'], price, timestamp, profit))
                    self.open_trades.pop(0)
                    self.position = 0
                    logger.info(f"执行平空操作: {trade_action}")
                    return trade_action
                else:
                    logger.warning("尝试平空，但没有开放的空头交易。正在重置持仓状态。")
                    self.position = 0

            # 开多
            self.position = 1
            self.open_trades.append({'price': price, 'size': contract_size, 'time': timestamp})
            self.trades.append(('开多', contract_size, price, timestamp, None, None, None))
            trade_action = {
                "action": "开多",
                "size": contract_size,
                "price": price,
                "timestamp": timestamp
            }
            logger.info(f"执行开多操作: {trade_action}")
            return trade_action

        elif signal == -1 and self.position >= 0:  # 卖出信号
            if self.position == 1:
                if self.open_trades:
                    # 平多
                    open_trade = self.open_trades[0]
                    profit_points = price - open_trade['price']
                    profit = profit_points * self.point_value - self.commission  # 只在平仓时扣除手续费
                    self.balance += profit
                    self.total_profit += profit
                    trade_action = {
                        "action": "平多",
                        "size": 1,
                        "price": price,
                        "timestamp": timestamp,
                        "profit": profit,
                        "open_price": open_trade['price'],
                        "open_time": open_trade['time']
                    }
                    self.trades.append(('平多', 1, open_trade['price'], open_trade['time'], price, timestamp, profit))
                    self.open_trades.pop(0)
                    self.position = 0
                    logger.info(f"执行平多操作: {trade_action}")
                    return trade_action
                else:
                    logger.warning("尝试平多，但没有开放的多头交易。正在重置持仓状态。")
                    self.position = 0

            # 开空
            self.position = -1
            self.open_trades.append({'price': price, 'size': -contract_size, 'time': timestamp})
            self.trades.append(('开空', contract_size, price, timestamp, None, None, None))
            trade_action = {
                "action": "开空",
                "size": contract_size,
                "price": price,
                "timestamp": timestamp
            }
            logger.info(f"执行开空操作: {trade_action}")
            return trade_action

        self.last_trade_index = self.current_index
        return None

    def format_trade_result(self, trade_action):
        result = None
        if trade_action['action'] in ['开多', '开空']:
            result = {
                "交易类型": "多" if trade_action['action'] == '开多' else "空",
                "手数": int(trade_action['size']),
                "开仓价格": round(float(trade_action['price']), 2),
                "开仓时间": trade_action['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                "平仓价格": "未平仓",
                "平仓时间": "未平仓",
                "盈利": "未实现",
                "手续费": "未扣除",  # 开仓时不扣除手续费
                "总盈利": round(self.total_profit, 2)
            }
            # 执行鼠标点击
            self.execute_mouse_click(trade_action['action'])
        elif trade_action['action'] in ['平多', '平空', '平多(止损/止盈)', '平空(止损/止盈)']:
            result = {
                "交易类型": "多" if trade_action['action'] in ['平多', '平多(止损/止盈)'] else "空",
                "手数": int(trade_action['size']),
                "开仓价格": round(float(trade_action['open_price']), 2),
                "开仓时间": trade_action['open_time'].strftime("%Y-%m-%d %H:%M:%S"),
                "平仓价格": round(float(trade_action['price']), 2),
                "平仓时间": trade_action['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                "盈利": round(float(trade_action['profit']), 2),
                "手续费": self.commission,  # 平仓时扣除总手续费
                "总盈利": round(self.total_profit, 2)
            }
            # 执行鼠标点击
            self.execute_mouse_click("平仓")
        
        if result:
            print(f"执行交易: {result['交易类型']}")
        return result

    def save_state(self):
        state = {
            'balance': self.balance,
            'position': self.position,
            'trades': self.trades,
            'open_trades': self.open_trades,
            'total_profit': self.total_profit,
            'current_index': self.current_index,
            'last_trade_index': self.last_trade_index
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, default=str)
        #print(f"交易状态已保存到 {self.state_file}")

    def load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            self.balance = state.get('balance', self.balance)
            self.position = state.get('position', self.position)
            self.trades = state.get('trades', self.trades)
            self.open_trades = state.get('open_trades', self.open_trades)
            self.total_profit = state.get('total_profit', 0)  # 使用 get 方法，如果键不存在则默认为 0
            self.current_index = state.get('current_index', self.current_index)
            self.last_trade_index = state.get('last_trade_index', self.last_trade_index)
            print(f"已从 {self.state_file} 加载交易状态")
            return True
        return False

    def execute_mouse_click(self, action):
        try:
            if action == "开多":
                pyautogui.moveTo(328, 1170, duration=0.5)
                pyautogui.click()
                print("执行开多操作")
            elif action == "开空":
                pyautogui.moveTo(461, 1169, duration=0.5)
                pyautogui.click()
                print("执行开空操作")
            elif action == "平仓":
                pyautogui.moveTo(601, 1171, duration=0.5)
                pyautogui.click()
                print("执行平仓操作")
            time.sleep(0.5)  # 增加延迟时间，确保鼠标移动和点击操作完成
        except Exception as e:
            print(f"执行鼠标点击时出错: {str(e)}")

    def get_real_time_data(self):
        url = "https://45.futsseapi.eastmoney.com/sse/114_c2411_qt?token=1101ffec61617c99be287c1bec3085ff"
        try:
            response = requests.get(url, timeout=20, stream=True)
            response.raise_for_status()
            content = ''
            
            for chunk in response.iter_content(chunk_size=1024):
                data = chunk.decode('utf-8')
                if data.find("{") == -1:
                    continue
            
                data = data[data.find("{"):data.rfind("}")+1]
                #print(data)  # 打印原始数据，用于调试
                d = json.loads(data)
                
                if 'qt' in d and 'p' in d['qt'] and 'utime' in d['qt']:
                    price = float(d['qt']['p'])
                    timestamp = datetime.fromtimestamp(d['qt']['utime'])
                    return price, timestamp
                
            return None, None
                
        except requests.Timeout:
            print('请求超时')
            return self.get_real_time_data()
        except requests.RequestException as e:
            print(f'请求错误: {e}')
            return None, None

    def run_real_time_trading(self):
        print("开始实时交易...")
        while True:
            price, timestamp = self.get_real_time_data()
            if price is not None and timestamp is not None:
                # 如果数据为空，重新加载数据
                if self.data is None or len(self.data) == 0:
                    self.load_data()
                    self.preprocess_data()
                
                action, trade_record, total_profit = self.real_time_trade(timestamp, price)
                if action:
                    print(json.dumps(action, ensure_ascii=False, indent=2))
                if trade_record:
                    print("最新交易记录:")
                    print(json.dumps(trade_record, ensure_ascii=False, indent=2))
                #print(f"当前总盈利: ${total_profit:.2f}")
                self.save_state()
            
            self.handle_commands()
            while self.paused:
                time.sleep(0.1)
                self.handle_commands()
            
            time.sleep(1)  # 每秒请求一次数据

    def format_trade_for_display(self, trade):
        trade_type, size, open_price, open_time, close_price, close_time, profit = trade
        return {
            "交易类型": trade_type,
            "手数": size,
            "开仓价格": round(float(open_price), 2),
            "开仓时间": open_time.strftime("%Y-%m-%d %H:%M:%S") if isinstance(open_time, datetime) else str(open_time),
            "平仓价格": round(float(close_price), 2) if close_price is not None else "未平仓",
            "平仓时间": close_time.strftime("%Y-%m-%d %H:%M:%S") if isinstance(close_time, datetime) else "未平仓",
            "盈利": round(float(profit), 2) if profit is not None else "未实现"
        }

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
        std_dev = excess_returns.std()
        epsilon = 1e-10  # 添加一个小的值以避免除以零
        if std_dev <= epsilon:
            return 0  # 如果标准差接近零，夏普比率无法计算，返回0
        return np.sqrt(252) * excess_returns.mean() / std_dev

class PerformanceAnalysis:
    @staticmethod
    def calculate_drawdowns(portfolio_values):
        hwm = np.maximum.accumulate(portfolio_values)
        epsilon = 1e-10  # 添加一个小的值以避免除以零
        drawdowns = (hwm - portfolio_values) / (hwm + epsilon)
        return drawdowns

    @staticmethod
    def calculate_max_drawdown(portfolio_values):
        drawdowns = PerformanceAnalysis.calculate_drawdowns(portfolio_values)
        return drawdowns.max()

    @staticmethod
    def calculate_cagr(initial_value, final_value, years):
        epsilon = 1e-10  # 添加一个小的值以避免除以零或负值
        if initial_value <= epsilon or final_value <= epsilon or years <= epsilon:
            return 0  # 如果输入值不合法，返回0
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

def run_trading_period(trading_system, start_date, end_date):
    trading_system.start_date = pd.to_datetime(start_date)
    trading_system.end_date = pd.to_datetime(end_date)
    trading_system.load_data()
    trading_system.preprocess_data()

    # 检查数据是否为空
    if trading_system.data.empty:
        print(f"警告：在时间段 {start_date} 到 {end_date} 内没有数据")
        return

    # 检查之前的交易，只保留在当前时间段之前的交易
    trading_system.trades = [trade for trade in trading_system.trades 
                             if pd.to_datetime(trade[3]) < trading_system.start_date]
    
    # 更新 current_index 为新时间段的起始位置
    start_data = trading_system.data[trading_system.data['datetime'] >= trading_system.start_date]
    if start_data.empty:
        print(f"警告：在开始日期 {start_date} 之后没有数据")
        return
    trading_system.current_index = start_data.index[0]

    # 模拟实时交易
    for i in range(trading_system.current_index, len(trading_system.data)):
        timestamp = trading_system.data['datetime'].iloc[i]
        if timestamp > trading_system.end_date:
            break
        price = trading_system.data['close'].iloc[i]
        
        action = trading_system.real_time_trade(timestamp, price)
        if action:
            print(json.dumps(action, ensure_ascii=False, indent=2))

    # ... (其余代码保持不变)

def main():
    csv_file = "e:/data/29#CL8.csv"
    initial_balance = 100000
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    state_file = 'trading_state.json'
    
    trading_system = TradingSystem(csv_file, initial_balance, start_date, end_date, state_file)
    
    # 启动键盘监听线程
    threading.Thread(target=trading_system.keyboard_listener, daemon=True).start()
    
    # 尝试加载之前的状态
    if trading_system.load_state():
        print("继续之前的交易")
        # 显示之前的交易记录
        print("\n=== 之前的交易记录 ===")
        for trade in trading_system.trades:
            trade_info = trading_system.format_trade_for_display(trade)
            print(json.dumps(trade_info, ensure_ascii=False, indent=2))
        print("=== 交易记录结束 ===\n")
    else:
        print("开始新的交易模拟")
    
    # 无论是否加载了状态，都重新加载和预处理数据
    trading_system.load_data()
    trading_system.preprocess_data()

    if trading_system.data is None or len(trading_system.data) == 0:
        print("错误：没有加载到交易数据")
        return

    # 运行实时交易
    trading_system.run_real_time_trading()

if __name__ == "__main__":
    main()