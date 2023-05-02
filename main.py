from tqdm import tqdm
import os
import random

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import analyze

def read_data():
    full_skill_df = pd.DataFrame()
    full_mini_df = pd.DataFrame()
    for username in tqdm(random.sample(os.listdir('./data'), 7500)):
        skill_df = pd.read_csv(f'./data/{username}/skill.csv')
        skill_df = skill_df.set_index('Skill')[['XP']].transpose()
        skill_df = np.log2(skill_df)
        skill_df.index = [username]
        full_skill_df = pd.concat([full_skill_df, skill_df])
        
        mini_df = pd.read_csv(f'./data/{username}/mini.csv')
        mini_df = mini_df.set_index('Minigame')[['Score']].transpose()
        mini_df.index = [username]
        full_mini_df = pd.concat([full_mini_df, mini_df])
    
    full_skill_df = full_skill_df.fillna(0)
    full_mini_df = full_mini_df.fillna(0)
    return full_skill_df, full_mini_df
    

def main():
    skill_df, mini_df = read_data()
    combat_skills = ['Attack', 'Defence', 'Strength', 'Hitpoints', 'Ranged', 'Prayer', 'Magic']
    
    random.seed(3781)
    # for minigame in mini_df.columns:
    #     selector = mini_df[minigame] > 0
    #     print(f"{minigame}:{len(skill_df.loc[selector])}")
    #     df = skill_df[combat_skills].loc[selector].join(mini_df[minigame].loc[selector], how='outer')
    #     if len(df) > 75:
    #         df = df.loc[random.sample(list(df.index), 75)]
    minigames = list(mini_df.columns)
    for label, data_df in zip(#['overall/skills', 'overall/minigames', 'overall/combat_only'] + 
                              [m+'_pop' for m in minigames],
                              #[skill_df, mini_df, skill_df[combat_skills]] +
                              [skill_df.join(mini_df[m], how='outer').loc[mini_df[m] > 0] for m in minigames]):
        print()
        print(label)
        all_lifetimes = []
        indices = list(range(len(data_df)))[:500]
        while len(indices) > 0:
            curr_idxs = random.sample(indices, min(25, len(indices)))
            indices = [x for x in indices if x not in curr_idxs]
            df = data_df.iloc[curr_idxs]
            
            fname = f'./figs/{label}'
            if not os.path.exists(fname):
                os.makedirs(fname)
        
            print("Setting up filtration with pre-computation...")
            filt = analyze.Filtration(df.to_numpy(),
                                    met_func=analyze.euc_metric, alpha=False)
            
            max_dim = 3
            print("Creating filtration...")
            euler_history = filt.filter(max_dim=max_dim)
            
            print("Calculating homologies...")
            new_lifetimes = [(life, [{list(df.index)[vert] for vert in face} for face in faces]) for life, faces in filt.compute_persistence()]
            all_lifetimes.extend(new_lifetimes)
            
            plt.plot([threshold for threshold in filt.thresholds], euler_history)
            plt.xlim((0, filt.thresholds[-1]))
            # plt.title(f"Euler Characteristic for '{minigame}'")
            plt.title(f"Euler Characteristic ({label})")
            plt.xlabel("Filtration Threshold")
            plt.savefig(f"{fname}/euler.png")
            plt.close()

            f_max = 0
            
            for dim in range(max_dim+1): 
                print(f"Calculating H{dim} persistence...")
                lifetimes = [(l, f) for l, f in all_lifetimes if (len(f[0]) - 1) == dim]
                with open(f"{fname}/lifetimes_{dim}.txt", 'w') as f:
                    for life, faces in sorted(lifetimes, key=lambda x: x[0][1] - x[0][0], reverse=True):
                        f_max = max(f_max, life[1])
                        if not np.allclose(life[0], life[1]):
                            f.write(f"{life}:{faces}\n")

                plt.scatter([x[0][0] for x in lifetimes], [y[0][1] for y in lifetimes], s=3)
                plt.xlabel("Birth")
                plt.ylabel("Death")
                plt.plot([0,f_max], [0,f_max], 'r--')

                plt.title(f"Persistent {dim}-Homologies ({label})")
                plt.xlim((-0.05*f_max,f_max))
                plt.ylim((-0.05*f_max,f_max))

                plt.savefig(f"{fname}/H{dim}_persistence.png")
                plt.close()


if __name__=='__main__':
    main()