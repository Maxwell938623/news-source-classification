import pandas as pd, re, html
from urllib.parse import urlparse

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

df = pd.read_csv('data/scraped/raw_scraped_headlines_merged.csv', dtype=str)
df = df[df['scrape_status'] == 'success'].copy()
df = df[df['raw_headline'].notna() & (df['raw_headline'].str.strip() != '')].copy()
df['headline_clean'] = df['raw_headline'].apply(clean)
df = df[df['headline_clean'].str.strip() != ''].copy()
df = df.drop_duplicates(subset=['headline_clean', 'source'])
df['section'] = df['url'].apply(extract_section)
df = df[df['source'].isin(['FoxNews', 'NBC'])].copy()
n_fox = len(df[df['source'] == 'FoxNews'])
n_nbc = len(df[df['source'] == 'NBC'])
print(f'Scraped total: {len(df)}  Fox={n_fox}  NBC={n_nbc}')

helper_fox_secs = ['politics','media','lifestyle','us','entertainment','world','sports',
                   'health','travel','opinion','food-drink','official-polls','tech',
                   'live-news','faith-values','science','auto','great-outdoors']
helper_nbc_secs = ['politics','news','select','tech','science','health','investigations',
                   'nbc-out','pop-culture','business','sports','weather','meet-the-press',
                   'think','feature','data-graphics','media','specials','shopping']

print('\nFox section availability in scraped data (helper sections only):')
fox = df[df['source'] == 'FoxNews']
for s in helper_fox_secs:
    n = len(fox[fox['section'] == s])
    print(f'  {s:<28} {n:>6}')

print('\nNBC section availability in scraped data (helper sections only):')
nbc = df[df['source'] == 'NBC']
for s in helper_nbc_secs:
    n = len(nbc[nbc['section'] == s])
    print(f'  {s:<28} {n:>6}')

# word count stats for scraped vs helper
df['word_count'] = df['headline_clean'].str.split().str.len()
print('\nScraped headline length:')
for src in ['FoxNews', 'NBC']:
    wc = df[df['source'] == src]['word_count']
    print(f'  {src}: mean={wc.mean():.1f}  median={wc.median():.0f}  std={wc.std():.1f}')
