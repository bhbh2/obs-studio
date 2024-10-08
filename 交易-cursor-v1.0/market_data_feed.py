import pandas as pd

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
