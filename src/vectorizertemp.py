from src.preprocess2 import *

from src.util import clean_route

x = load_dataset('train.xlsx')
y = load_dataset('test.xlsx')
ms = load_dataset('ms.json')['market']
bk_class = load_dataset('class.json')['class']

Feat = [
    ('Base Features', BaseFeat(ms, bk_class)),
    ('Log Transform', LogTransformer(['Duration']))  # ,
    # ('Vectoriser', CustomVectoriser(['Route']))
]

pipe_feat = Pipeline(Feat)
df = pipe_feat.transform(x)
df['Route'] = df['Route'].apply(clean_route)

tf = TfidfVectorizer(ngram_range=(1, 1), lowercase=False)
# out = tf.fit(df['Route'])

train_route = tf.fit_transform(df['Route'])
train_route = pd.DataFrame(data=train_route.toarray(), columns=tf.get_feature_names())
df = pd.concat([df, train_route], axis=1)
df.drop('Route', axis=1, inplace=True)

df_test = pipe_feat.transform(y)
df_test['Route'] = df_test['Route'].apply(clean_route)
test_route = tf.fit_transform(df_test['Route'])
test_route = pd.DataFrame(data=test_route.toarray(), columns=tf.get_feature_names())
df_test = pd.concat([df_test, test_route], axis=1)
df_test.drop('Route', axis=1, inplace=True)
train_df = df
test_df = df_test

train_df = pd.get_dummies(train_df, columns=['Airline', 'Source', 'Destination', 'Additional_Info', 'Date_of_Journey',
                                             'Dep_Time', 'Arrival_Time', 'Dep_timeofday', 'Booking_Class',
                                             'Arr_timeofday'],
                          drop_first=True)
test_df = pd.get_dummies(test_df, columns=['Airline', 'Source', 'Destination', 'Additional_Info', 'Date_of_Journey',
                                           'Dep_Time', 'Arrival_Time', 'Dep_timeofday', 'Booking_Class',
                                           'Arr_timeofday'],
                         drop_first=True)
