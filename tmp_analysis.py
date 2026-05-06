import pandas as pd, numpy as np, re, html, collections
from urllib.parse import urlparse

df = pd.read_csv('helpers/url_with_headlines.csv', dtype=str)

def clean(t):
    t = html.unescape(str(t))
    t = re.sub(r'<[^>]+>', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

def infer_source(url):
    u = str(url).lower()
    if 'foxnews.com' in u: return 'FoxNews'
    if 'nbcnews.com' in u: return 'NBC'
    return 'Other'

def extract_section(url):
    path = urlparse(str(url)).path.strip('/')
    return path.split('/')[0].lower() if path else '_root'

df['source']  = df['url'].apply(infer_source)
df['section'] = df['url'].apply(extract_section)
df['headline_clean'] = df['headline'].apply(clean)
df['word_count'] = df['headline_clean'].str.split().str.len()
df = df[df['source'] != 'Other'].copy()

n_fox = len(df[df['source'] == 'FoxNews'])
n_nbc = len(df[df['source'] == 'NBC'])
print(f'Total: {len(df)}  Fox={n_fox}  NBC={n_nbc}')

print('\n--- SOURCE DISTRIBUTION ---')
print(df['source'].value_counts(normalize=True).round(4).to_string())

print('\n--- HEADLINE LENGTH (word count) ---')
for src in ['FoxNews', 'NBC']:
    wc = df[df['source'] == src]['word_count']
    print(f'{src}: mean={wc.mean():.1f}  median={wc.median():.0f}  std={wc.std():.1f}  min={wc.min()}  max={wc.max()}')

print('\n--- SECTION DISTRIBUTION per source ---')
for src in ['FoxNews', 'NBC']:
    sub = df[df['source'] == src]
    sc = sub['section'].value_counts()
    sc_pct = (sc / len(sub) * 100).round(2)
    print(f'\n{src} ({len(sub)} rows):')
    for sec, cnt in sc.items():
        print(f'  {sec:<32} {cnt:>5}  ({sc_pct[sec]:.2f}%)')

print('\n--- SECTION OVERLAP ---')
fox_secs = set(df[df['source'] == 'FoxNews']['section'].unique())
nbc_secs = set(df[df['source'] == 'NBC']['section'].unique())
print('Only Fox:', sorted(fox_secs - nbc_secs))
print('Only NBC:', sorted(nbc_secs - fox_secs))
print('Shared  :', sorted(fox_secs & nbc_secs))

print('\n--- TOP WORDS by source ---')
from collections import Counter
stop = {'the','a','an','in','of','to','and','for','on','with','is','as','at',
        'by','his','her','its','their','after','from','that','this','was','are'}
for src in ['FoxNews', 'NBC']:
    words = []
    for h in df[df['source'] == src]['headline_clean']:
        words += [w.lower() for w in str(h).split() if w.lower() not in stop and len(w) > 2]
    top = Counter(words).most_common(20)
    print(f'\n{src} top words: {[w for w,c in top]}')

print('\n--- WORD COUNT BUCKETS ---')
bins = [0, 8, 11, 14, 17, 100]
labels = ['<=8', '9-11', '12-14', '15-17', '18+']
df['len_bucket'] = pd.cut(df['word_count'], bins=bins, labels=labels)
print(pd.crosstab(df['source'], df['len_bucket'], normalize='index').round(3).to_string())
