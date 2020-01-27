from src.preprocess2 import *


x = load_dataset('train.xlsx')
ms = load_dataset('ms.json')['market']
bk_class = load_dataset('class.json')['class']

Feat = [
        ('Base Features', BaseFeat(ms, bk_class)),
        ('Log Transform', LogTransformer(['Price', 'Duration']))#,
        #('Vectoriser', CustomVectoriser(['Route']))
    ]

pipe_feat = Pipeline(Feat)
df = pipe_feat.transform(x)

tf = TfidfVectorizer(ngram_range=(1, 1), lowercase=False)
#out = tf.fit(df['Route'])

train_route = tf.fit_transform(df['Route'])
train_route = pd.DataFrame(data=train_route.toarray(), columns=tf.get_feature_names())
df = pd.concat([df, train_route], axis=1)
df.drop('Route', axis=1, inplace=True)