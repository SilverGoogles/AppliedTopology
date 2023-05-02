import pandas as pd
import requests
import time
import random
from tqdm import tqdm
import urllib
import os
import ast

url = 'https://secure.runescape.com/m=hiscore_oldschool/'

header = {
  "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246",
  "X-Requested-With": "XMLHttpRequest"
}

def get_random_scores(timeout=3600):
    start = time.time()
    pages = list(range(1,80000+1))
    random.shuffle(pages)
    for page in tqdm(pages):
        while True:
            try:
                r = requests.get(url + f"overall?table=0&page={page}", headers=header)
                break
            except requests.exceptions.ConnectionError:
                time.sleep(10)
                continue
        try:
            dfs = pd.read_html(r.text)[0]
        except ValueError as e:
            print(e)
            time.sleep(60*2)
            continue
        usernames = list(dfs.iloc[1:]['Name'])
        get_scores(usernames)
        
        if (time.time() - start) > timeout:
            break
        

def get_scores(usernames):
    level_dict = dict()
    for user in usernames:
        name = urllib.parse.quote(user).replace('%C2', '')
        if os.path.exists(f'./data/{name}'):
            print(f"Data for {name} exists, skipping...")
            continue
        
        
        while True:
            try:
                r = requests.get(url + f'hiscorepersonal?user1={name}', headers=header)
                break
            except requests.exceptions.ConnectionError:
                time.sleep(10)
                continue
        try:
            dfs = pd.read_html(r.text)
        except ValueError as e:
            print(e)
            time.sleep(60*2)
            continue
        
        if len(dfs) != 3:
            continue
        dfs = dfs[2]
        dfs = dfs.loc[[row for row in dfs.index if not dfs.loc[row].isnull().all()]]
        dfs.index = range(len(dfs))
        
        mini_idx = dfs.index[dfs[0] == 'Minigame']
        mini_df = dfs.loc[mini_idx[0]:, 1:]
        mini_df.columns = mini_df.iloc[0]
        mini_df.columns.name = ''
        mini_df = mini_df.iloc[1:]
        mini_df = mini_df.iloc[:, 0:3]
        mini_df.index = mini_df['Minigame']
        mini_df = mini_df[['Rank', 'Score']]
        
        skill_idx = dfs.index[dfs[0] == 'Skill']
        skill_df = dfs.loc[skill_idx[0]:mini_idx[0]-1, 1:]
        skill_df.columns = skill_df.iloc[0]
        skill_df.columns.name = ''
        skill_df = skill_df.iloc[1:]
        skill_df.index = skill_df['Skill']
        skill_df = skill_df[['Rank', 'Level', 'XP']]
        
        total_level = skill_df.loc['Overall', 'Level']
        if total_level not in level_dict:
            level_dict[total_level] = []
        level_dict[total_level].append(name)
        
        os.makedirs(f'./data/{name}')
        skill_df.to_csv(f'./data/{name}/skill.csv')
        mini_df.to_csv(f'./data/{name}/mini.csv')
        
def main():
    get_scores(['Pwofesor0w0k', 'ChiefSt0rm', 'hom0erotic'])
    get_random_scores(timeout=60*60*30)
    
if __name__=='__main__':
    main()
    