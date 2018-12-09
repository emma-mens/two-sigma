import pickle
import gc
import time
import ast
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler

NEWS_FEATURES = ['urgency', 'takeSequence','bodySize', 'companyCount', 'marketCommentary',
                 'sentenceCount', 'wordCount','firstMentionSentence', 'relevance', 'sentimentClass',                      'sentimentNegative', 'sentimentNeutral', 'sentimentPositive','sentimentWordCount',
                 'noveltyCount5D', 'noveltyCount7D', 'volumeCounts24H', 'volumeCounts3D', 'month']

MARKET_FEATURES = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
                   'returnsClosePrevMktres1', 'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 
                   'returnsOpenPrevRaw10', 'returnsClosePrevMktres10', 'returnsOpenPrevMktres10',
                   'universe']

LABEL_COL = ['returnsOpenNextMktres10']

class FinanceDataset(Dataset):
    def __init__(self, data_path, augmented_data_path, single_df=False, 
                 label_col=LABEL_COL,
                 market_features=MARKET_FEATURES,
                 news_features=NEWS_FEATURES, 
                 data_type='val', max_interval='7 days', returns_direction=False):
        """ 
        data_type in {train|val|test}
        """
        print('loading data')
        start_ = time.time()
        if single_df:
            with open(augmented_data_path, 'rb') as f:
                market_train = pickle.load(f)
            market_train = pd.concat([market_train, pd.get_dummies(market_train['provider'])], axis=1)
            market_train = market_train.drop(columns=['provider','sourceTimestamp', 'firstCreated', 'headlineTag'])
            self.market_features = [i for i in market_train.columns if i not in label_col]
            split = int(0.8*market_train.shape[0])
            if data_type == 'train':
                self.market = market_train[:split]
            elif data_type == 'val':
                self.market = market_train[split:]
            else:
                print('Provide a df with time column to test')
                exit()
        else:
            with open(data_path, 'rb') as f:
                train, test = pickle.load(f)

            _, news_train = train
            market_train = pd.read_csv(augmented_data_path)
            market_train = market_train[market_train.ndays == max_interval]

            news_train['date'] = pd.DatetimeIndex(news_train.time).normalize()
            news_train['month'] = news_train.date.dt.month
            market_train['date'] = pd.DatetimeIndex(market_train.time).normalize().tz_localize('UTC')
            self.news_features = news_features
            self.market_features = market_features
            
            # split train into train, val
            test_start = market_train.date.max() - pd.DateOffset(years=1)
            val_start = test_start - pd.DateOffset(years=1)

            if data_type == 'train':
                start = market_train.date.min(); end = val_start
            elif data_type == 'val':
                start = val_start; end = test_start
            else:
                start = test_start; end = market_train.date.max()

            self.max_interval = pd.Timedelta(max_interval)
            self.market = market_train[(market_train.date >= start) & (market_train.date < end)]
        print('done loading data; took', time.time() - start_, 'seconds')

        self.label_col = label_col
        
        if not single_df:
            self.news = news_train[(news_train.date >= start - self.max_interval) &
                               (news_train.date < end - self.max_interval)]
            self.news_len = len(self.news)
        
        if returns_direction:
            self.market.returnsOpenNextMktres10 = (self.market.returnsOpenNextMktres10 > 0).apply(int)
         
        # remove outliers in returns
        self.market = self.market[(self.market.returnsOpenNextMktres10.values > -1) &
                                  (self.market.returnsOpenNextMktres10.values < 1)]
        print('scaling data')
        start_ = time.time()
        self.market_scaler = MinMaxScaler()
        
        if not single_df:
            self.news_scaler = MinMaxScaler()
            news_numeric = self.news[self.news_features].copy()
            news_numeric = news_numeric.dropna()

            news = self.news.copy()

            news[self.news_features] = self.news_scaler.fit_transform(news_numeric)
            self.news = news
            del news_numeric
            del news
            del news_train
            gc.collect()
        
        market_numeric = self.market[self.market_features + self.label_col].copy()
        market_numeric = market_numeric.dropna()
        market = self.market.copy()
        
        market[self.market_features + self.label_col] = self.market_scaler.fit_transform(market_numeric)
        self.market = market
        del market_numeric
        del market
        gc.collect()
        
        print('done scaling data; took', time.time() - start_, 'seconds')
                
        del market_train
        gc.collect()
        
        if not single_df:
            self.market.indices = self.market.indices.apply(ast.literal_eval).apply(np.array)
        self.single_df = single_df
        
    def __len__(self):
        return len(self.market)
    
    def __getitem__(self, idx):

        asset = self.market.iloc[idx]
        
        market_data = asset[self.market_features].values.reshape((1,-1))
            
        if self.single_df:
            data = market_data.astype(np.float32)
        else:
            # don't index out of news df
            news_asset = self.news.iloc[asset.indices[asset.indices < self.news_len]] 
            news_asset = news_asset[news_asset.time > (asset.date - self.max_interval)]

            # For now, we're using only 10 articles [selection process to be justified later]
            n_articles = 10
            data = news_asset[self.news_features][-n_articles:].values
            rows,col = data.shape
            if rows < n_articles:
                data = np.vstack((data, np.zeros((n_articles - rows, col))))
            data = data.reshape((1,-1))
            data = np.hstack((market_data, data)).astype(np.float32)
        
        data = torch.FloatTensor(data)
        label = torch.FloatTensor(asset[self.label_col])
        return data, label
        
    def _get_news_in_time_range(self, news, time, ndays):
        return self.news[(self.news.time <= time) & 
                         (self.news.time >= time-pd.to_timedelta(ndays))]
        
    def _get_news_for_batch(self, market_time_start, market_time_end, ndays='2 days'):
        return self.news[(self.news.time <= market_time_end) & 
                         (self.news.time >= market_time_start - pd.to_timedelta(ndays))]