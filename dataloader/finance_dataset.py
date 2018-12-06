import pickle
import gc
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class FinanceDataset(Dataset):
    def __init__(self, data_path, data_type='val', max_interval='2 days'):
        """ 
        data_type in {train|val|test}
        """
        print('loading data')
        start_ = time.time()
        with open(data_path, 'rb') as f:
            train, test = pickle.load(f)
            
        market_train, news_train = train
        print('done loading data; took', time.time() - start_, 'seconds')
        
        news_train['date'] = pd.DatetimeIndex(news_train.time).normalize()
        news_train['month'] = news_train.date.dt.month
        market_train['date'] = pd.DatetimeIndex(market_train.time).normalize()
        
        self.news_features = ['urgency', 'takeSequence','bodySize', 'companyCount', 'marketCommentary',
                         'sentenceCount', 'wordCount','firstMentionSentence', 'relevance', 
                         'sentimentClass', 'sentimentNegative', 'sentimentNeutral', 
                         'sentimentPositive','sentimentWordCount', 'noveltyCount5D', 
                         'noveltyCount7D', 'volumeCounts24H', 'volumeCounts3D', 'month']
        
        self.market_features = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
                           'returnsClosePrevMktres1', 'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 
                           'returnsOpenPrevRaw10', 'returnsClosePrevMktres10', 'returnsOpenPrevMktres10', 'universe']
        self.label_col = ['returnsOpenNextMktres10']
        
        # split train into train, val
        test_start = market_train.date.max() - pd.DateOffset(years=1)
        val_start = test_start - pd.DateOffset(years=1)

        if data_type == 'train':
            start = market_train.date.min(); end = val_start
        elif data_type == 'val':
            start = val_start; end = test_start
        else:
            start = test_start; end = market_train.date.max()
            
        self.market_index_offset = len(market_train[(market_train.date < start)].index)
        self.news_index_offset = len(news_train[(news_train.date < start - pd.Timedelta(max_interval))])
        self.market = market_train[(market_train.date >= start) & (market_train.date < end)]
        self.news = news_train[(news_train.date >= start - pd.Timedelta(max_interval)) &
                              (news_train.date < end - pd.Timedelta(max_interval))]
        
        print('scaling data')
        start_ = time.time()
        self.news_scaler = StandardScaler()
        self.market_scaler = StandardScaler()
        
        news_numeric = self.news[self.news_features].copy()
        news_numeric = news_numeric.fillna(0)
        
        news = self.news.copy()
        
        news[self.news_features] = self.news_scaler.fit_transform(news_numeric)
        self.news = news
        del news_numeric
        del news
        gc.collect()
        
        market_numeric = self.market[self.market_features + self.label_col].copy()
        market_numeric = market_numeric.fillna(0)
        market = self.market.copy()
        
        market[self.market_features + self.label_col] = self.market_scaler.fit_transform(market_numeric)
        self.market = market
        del market_numeric
        del market
        gc.collect()
        
        print('done scaling data; took', time.time() - start_, 'seconds')
        
        self.max_interval = max_interval
        
        self.prev_idx = self.market_index_offset
        self.market_batch = 100
        
        del market_train
        del news_train
        gc.collect()
        
    def __len__(self):
        return len(self.market)
    
    def __getitem__(self, idx):
        index = idx + self.market_index_offset
        asset = self.market.loc[index]
        
        # update chunk of news to look within
        j = idx % self.market_batch
        if j == 0:
            market = train_loader.dataset.market[self.market_batch*j: self.market_batch*(j+1)]
            self.current_news = self._get_news_for_batch(market.time.min(), 
                                                         market.time.max(), self.max_interval)
            
        news_asset = self.current_news[self.current_news.assetName == asset.assetName]
        
        # For now, we're using only 3 articles [selection process to be justified later]
        n_articles = 3
        data = news_asset[self.news_features][:n_articles].values
        rows,col = data.shape
        if rows < n_articles:
            data = np.vstack((data, np.zeros((n_articles - rows, col))))
        data = data.reshape((-1,1))
        data = np.vstack((asset[self.market_features].values.reshape((-1,1)), data)).astype(np.float32)
        data = torch.FloatTensor(data)
        label = torch.FloatTensor(asset[self.label_col])
        return data, label
        
    def _get_news_in_time_range(self, news, time, ndays):
        return self.news[(self.news.time <= time) & 
                         (self.news.time >= time-pd.to_timedelta(ndays))]
        
    def _get_news_for_batch(self, market_time_start, market_time_end, ndays='2 days'):
        return self.news[(self.news.time <= market_time_end) & 
                         (self.news.time >= market_time_start - pd.to_timedelta(ndays))]