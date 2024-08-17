import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import fbeta_score, make_scorer



df = pd.read_csv("all_train20240814.csv")
# Define the features and target
features=list(df.columns)
target = 'popular'
feature.remove(target)
X= df[features]
y = df[target]


# Handling numerical,categorical and text features separately


# Create a column transformer for numerical features
class Numeric_Transformer(object):
    """
    transfer the data to numeric
    including['price', 'promotion', 'Unit Count_comb']
    """
    
    def fit(self, X, y=None):
        df = pd.DataFrame()
        df['Price'] = X.price.map(self.price2num)
        self.pric_mean = df['Price'].mean()

        df['Unit Count'] = X['Unit Count_comb'].map(self.count_num)
        self.count_mean = df['Unit Count'].mean()
        
    def transform(self, X, y=None):
        df = pd.DataFrame()
        df['Price'] = X.price.map(self.price2num)
        df['Price'].fillna(self.pric_mean, inplace=True)
        
        X['promotion'] = X['promotion'].replace(
            {
                '1 applicable promotion':'1 Applicable Promotion',
                '2 applicable promotion(s)':'2 Applicable Promotion(s)',
            }

        )
        df['Promotion'] = X.promotion.map(self.promot_to_num)

        df['Unit Count'] = X['Unit Count_comb'].map(self.count_num)
        df['Unit Count'].fillna(self.count_mean, inplace=True)
        return df
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def price2num(self, x):
        """
        This function transforms a price string to a numerical value.
        """
        if isinstance(x, str):
            if len(x) > 8:
                return np.nan
            elif 'S$' in x:
                x = x.replace('S$', '').strip()
                return float(x) * 0.74
            else:
                x = x.replace('$', '').strip()
                return float(x) * 1.00
        return x

    def promot_to_num(self, value):
        """
        This function transforms a promotion string to a numerical value.
        """
        if pd.isna(value):
            return 0
        elif '1 Applicable Promotion' in value:
            return 1
        elif '2 Applicable Promotion(s)' in value:
            return 2
        elif '3 Applicable Promotion(s)' in value:
            return 3
        else:
            return 0  # or handle unexpected values

    def count_num(self, text):
        """
        This function transforms a 'unit count' string to a numerical value.
        """
        if type(text) == str:
            text = text.split()[0]
            # there are some mixing values
            if text in ['Softgel', 'Liquid']:
                return None
            return float(text)

# Create a column transformer for categorical features
class Categorical_Transformer_Freq(object):
    """
    transfer the data to numeric
    including ['Brand', 'Manufacturer', 'Primary Supplement Type_comb', 'Age Range_comb','Flavor_comb', 'Item_Form_updt', 'Diet Type_comb']
    """

    def fit(self, X, y=None):
        """
        Fit the encoder by calculating frequencies of the values in the column.
        """
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        # Compute 'Brand' frequency counts
        #{3(high frequency): >10, 2(mid frequency): [2,11], 1(low frequency): 1, 0: null}
        self.brand_freq = X['Brand'].value_counts()
        self.brand_high = self.brand_freq[self.brand_freq > 10].index.tolist()
        self.brand_mid = self.brand_freq[(self.brand_freq > 1) & (self.brand_freq <= 11)].index.tolist()
        self.brand_low = self.brand_freq[self.brand_freq == 1].index.tolist()

        # Compute 'Manufacturer' frequency counts
        # {3:high freq, 2: mid freq, 1: low freq, 0: null}
        self.Manufacturer_freq = X['Manufacturer'].value_counts()
        self.Manufacturer_high = self.Manufacturer_freq[self.Manufacturer_freq >= 5].index.tolist()
        self.Manufacturer_mid = self.Manufacturer_freq[(self.Manufacturer_freq > 1) & (self.Manufacturer_freq < 5)].index.tolist()
        self.Manufacturer_low = self.Manufacturer_freq[self.Manufacturer_freq == 1].index.tolist()

        # Compute 'flavor' frequency counts
        # {3:high freq, 2: mid freq, 1: low freq, 0: null}
        X['Flavor_comb'] = X['Flavor_comb'].replace(
            {
                'Unflavored': None,
                'Unflavoured':None,
            }
        )
        self.flavor_freq = X['Flavor_comb'].value_counts()
        self.flavor_high = self.flavor_freq[self.flavor_freq >= 5].index.tolist()
        self.flavor_mid = self.flavor_freq[(self.flavor_freq > 1) & (self.flavor_freq < 5)].index.tolist()
        self.flavor_low = self.flavor_freq[self.flavor_freq == 1].index.tolist()

        # Compute 'Primary Supplement Type_comb' frequency counts
        # {3:high freq, 2: mid freq, 1: low freq, 0: null}
        self.supp_freq = X['Primary Supplement Type_comb'].value_counts()
        self.supp_high = self.supp_freq[self.supp_freq >= 10].index.tolist()
        self.supp_mid = self.supp_freq[(self.supp_freq > 1) & (self.supp_freq < 10)].index.tolist()
        self.supp_low = self.supp_freq[self.supp_freq == 1].index.tolist()

        ## Compute 'Primary Supplement Type_comb' frequency counts
        ## encoding: {2: common format, 1: seldom format, 0: no format infor}
        self.format_high = ['capsule', 'softgel', 'tablet', 'powder', 'gummy', 'lozenge']  

        # Compute 'age' frequency counts
        # {3:high freq, 2: mid freq, 1: low freq, 0: null}
        X['Age Range_comb'] = X['Age Range_comb'].replace(
            {
                'Adult,Kid':'All Ages',
                'All ages':'All Ages',
                'All Stages':'All Ages',
                'Adult,Senior': 'Adult',
                'Adult,Baby':'All Ages',
                'Over 18s': 'Adult',
                'Adults': 'Adult',
                'adult, teen': 'Adult,Teen',
                'Adult; Kid': 'over 6 years',
            }

        )
        self.age_dict = {'Adult': 0.8, 'Child ':0.2, 'All Ages':1, 'Teen':0.2, 'over 6 years':0.9, 'Adult,Teen':0.85, 'Above 3 years old':0.95, 'Baby':0.05, 'None':0}

        # Compute 'Diet Type' frequency counts
        # {3:high freq, 2: mid freq, 1: low freq, 0: null}
        self.diet_freq = X['Diet Type_comb'].value_counts()
        self.diet_high = self.diet_freq[self.diet_freq >= 10].index.tolist()
        self.diet_mid = self.diet_freq[(self.diet_freq > 1) & (self.diet_freq < 10)].index.tolist()
        self.diet_low = self.diet_freq[self.diet_freq == 1].index.tolist()
        
    def transform(self, X, y=None):
        """
        Transform the DataFrame by encoding the 'brand' column based on frequency.
        """
        df = pd.DataFrame()
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        df['Brand'] = X.Brand.map(self.brand2num)
        df['Manufacturer'] = X.Manufacturer.map(self.manu2num)
        df['Flavor'] = X.Flavor_comb.map(self.flavor2num)
        df['Primary_Supplement'] = X['Primary Supplement Type_comb'].map(self.supp2num)
        df['Item_Form'] = X.Item_Form_updt.map(self.form2num)
        Age = X['Age Range_comb'].replace(
            {
                'Adult,Kid':'All Ages',
                'All ages':'All Ages',
                'All Stages':'All Ages',
                'Adult,Senior': 'Adult',
                'Adult,Baby':'All Ages',
                'Over 18s': 'Adult',
                'Adults': 'Adult',
                'adult, teen': 'Adult,Teen',
                'Adult; Kid': 'over 6 years',
            }
        )
        df['Age_Range'] = Age.map(self.age2num)
        df['Diet_Type'] = X['Diet Type_comb'].map(self.diet2num)
        return df
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def brand2num(self, name):
        if name in self.brand_high: return 3
        elif name in self.brand_mid: return 2
        elif name in self.brand_low: return 1
        else: return 0

    def manu2num(self, name):
        if name in self.Manufacturer_high: return 3
        elif name in self.Manufacturer_mid: return 2
        elif name in self.Manufacturer_low: return 1
        else: return 0

    def flavor2num(self, name):
        if name in self.flavor_high: return 3
        elif name in self.flavor_mid: return 2
        elif name in self.flavor_low: return 1
        elif name in ['Unflavored', 'Unflavoured']: return 0
        else: return 0

    def supp2num(self, name):
        if name in self.supp_high: return 3
        elif name in self.supp_mid: return 2
        elif name in self.supp_low: return 1
        else: return 0

    def form2num(self, names):
        if type(names) == float: return 0
        else:
            names = names.replace('[','')
            names = names.replace(']','')
            names = names.replace("'",'')
            #print(names, type(names))
            if names in self.format_high: 
                return 2
            elif names: 
                return 1
            else: return 0

    def age2num(self, name):
        if name:
            if name in self.age_dict.keys():
                return self.age_dict[name]
            else:
                return 0.1
        else:
            return 0

    def diet2num(self, name):
        if name in self.diet_high: return 3
        elif name in self.diet_mid: return 2
        elif name in self.diet_low: return 1
        else: return 0

class Categorical_Transformer_other(object):
    """
    transfer the data to numeric data
    'Directions_updt' by binary transfer
    'Description_comb' by the total number of words
    """
    def fit(self, X, y=None):
        df = pd.DataFrame()
        df['Directions'] = X.Directions_updt.map(self.binary_convert)
        df['Description'] = X.Description_comb.map(self.count_words)
        
    def transform(self, X, y=None):
        df = pd.DataFrame()
        df['Directions'] = X.Directions_updt.map(self.binary_convert)
        df['Description'] = X.Description_comb.map(self.count_words)
        return df
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def binary_convert(self, content):
        """
        if there are contents, return 1;
        otherwise, return 0
        """
        if type(content) == str:
            return 1
        else: return 0

    def count_words(self, sentence):
        if type(sentence) == str:
            words = sentence.split()
            return len(words)
        else: return 0


# Text preprocessing pipeline
class Text_Transformer(object):
    """
    transfer the data 'title','Benefit_comb', 'Ingredient_comb' by NLP
    """
    def fit(self, X, y=None):
        X['title'] = X['title'].fillna('')
        # define exclude words
        self.title_exclude_words = ['mg','mcg']
       
        # extract the key words from 'title' and add to dataframe
        self.title_extraction = self.extract_keywords_tfidf(X['title'], self.title_exclude_words)
        self.title_extraction = pd.DataFrame(self.title_extraction, columns = ['Title_e'])
        #print(self.title_extraction)
        
        # Initialize CountVectorizer with binary=True for one-hot encoding
        self.vectorizer_title = CountVectorizer()
        
        # Fit and transform the text data
        self.vectorizer_title.fit(self.title_extraction['Title_e'])


        X['Benefit_comb'] = X['Benefit_comb'].fillna('')
        # define exclude words
        self.benefit_exclude_words = ['support', 'supports', 'health', 'healthy', 'supplement']
       
        # extract the key words from 'benefit' and add to dataframe
        self.benefit_extraction = self.extract_keywords_tfidf(X['Benefit_comb'], self.benefit_exclude_words)
        self.benefit_extraction = pd.DataFrame(self.benefit_extraction, columns = ['Benefit'])
        #print(self.benefit_extraction)
        
        # Initialize CountVectorizer with binary=True for one-hot encoding
        self.vectorizer_benefit = CountVectorizer()
        
        # Fit and transform the text data
        self.vectorizer_benefit.fit(self.benefit_extraction['Benefit'])
        
        X['Ingredient_comb'] = X['Ingredient_comb'].fillna('')
        self.ingr_exclude_words = ['mg', 'ingredients', 'blend', 'capsule','extract', 'powder']
        # extract the key words from 'Ingrdient' and add to dataframe
        self.ingr_extraction = self.extract_keywords_tfidf(X['Ingredient_comb'], self.ingr_exclude_words)
        self.ingr_extraction = pd.DataFrame(self.ingr_extraction, columns = ['Ingrdient'])
        #print(self.ingr_extraction)
        
        # Initialize CountVectorizer with binary=True for one-hot encoding
        self.vectorizer_ingr = CountVectorizer()
        
        # Fit and transform the text data
        self.vectorizer_ingr.fit(self.ingr_extraction['Ingrdient'])
        
    def transform(self, X, y=None):
        benefit_df = pd.DataFrame()
        ingr_df = pd.DataFrame()
        idx = X.index.tolist()

        X['Benefit_comb'] = X['Benefit_comb'].fillna('')
        # Fit and transform the text data
        benefit = self.vectorizer_benefit.transform(X['Benefit_comb'])
        # Convert the result to a DataFrame
        benefit_df = pd.DataFrame(benefit.toarray(), columns=self.vectorizer_benefit.get_feature_names_out(), index=idx)

        X['Ingredient_comb'] = X['Ingredient_comb'].fillna('')
        # Fit and transform the text data
        ingr = self.vectorizer_ingr.transform(X['Ingredient_comb'])
        # Convert the result to a DataFrame
        ingr_df = pd.DataFrame(ingr.toarray(), columns=self.vectorizer_ingr.get_feature_names_out(), index=idx)

        if not benefit_df.empty and not ingr_df.empty: return pd.concat([benefit_df, ingr_df], axis=1)
        elif benefit_df.empty and not ingr_df.empty: return benefit_df
        elif not benefit_df.empty and ingr_df.empty: return ingr_df
        else: return None

    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


    def extract_keywords_tfidf(self, column_data, exclude_words):
        # intiate TfidfVectorizer
        vectorizer = TfidfVectorizer(stop_words='english', max_features=20)  # increase max_features to make sure can get the key words 
        
        # train TF-IDF model and transform text 
        tfidf_matrix = vectorizer.fit_transform(column_data)
        
        # get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # get weight score for TF-IDF
        tfidf_scores = tfidf_matrix.toarray()
        
        # get keywords list
        keywords_list = []
        for row in tfidf_scores:
            top_keywords = [feature_names[i] for i in row.argsort()[-10:][::-1] if feature_names[i] not in exclude_words]
            keywords_list.append(', '.join(top_keywords[:5]))  # only keep the first 5 keywords
        
        return keywords_list

# Combine numerical, categorical, and text pipelines
class Combine_Transformer(object):
    def __init__(self):
        self.tsf = Numeric_Transformer()
        self.tsf1 = Categorical_Transformer_Freq()
        self.tsf2 = Categorical_Transformer_other()
        self.tsf3 = Text_Transformer()

    def fit(self, X, y=None):
        # Assuming X is a DataFrame with columns relevant to each transformer
        self.tsf.fit(X)
        self.tsf1.fit(X)
        self.tsf2.fit(X)
        self.tsf3.fit(X)
        return self

    def transform(self, X):
        # Transform each feature and concatenate the results
        transformed1 = self.tsf.transform(X[['price', 'promotion', 'Unit Count_comb']])
        transformed2 = self.tsf1.transform(X[['Brand', 'Manufacturer', 'Primary Supplement Type_comb', 'Age Range_comb','Flavor_comb', 'Item_Form_updt','Diet Type_comb']])
        transformed3 = self.tsf2.transform(X[['Directions_updt', 'Description_comb']])
        transformed4 = self.tsf3.transform(X[['title','Benefit_comb', 'Ingredient_comb']])
        
        # Concatenate the transformed features
        #print(transformed1.index)
        #print(transformed2.index)
        #print(transformed3.index)
        #print(transformed4.index)
        return np.hstack([transformed1, transformed2, transformed3, transformed4])

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
        
# Split the data
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert target labels to integer type if they are strings
#y_train = y_train.astype(int)
#y_test = y_test.astype(int) 

y = y.astype(int) 

# Define a pipeline that includes preprocessing and the model

model= Pipeline(steps=[
    ('tsf',Combine_Transformer()),
    ('rsc', MinMaxScaler()),
    ('classifier', xgb.XGBClassifier(colsample_bytree= 1.0, 
                                     learning_rate= 1.0, 
                                     max_depth= 3,
                                     min_child_weight= 3, 
                                     n_estimators=20, 
                                     scale_pos_weight= 4, 
                                     subsample=0.8,
                                    random_state=42
))
])

# Train the model
model.fit(X, y)

# Evaluate the model
#y_train_pred = XGBmodel_t.predict(X_train)
#y_test_pred = XGBmodel_t.predict(X_test)




with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
