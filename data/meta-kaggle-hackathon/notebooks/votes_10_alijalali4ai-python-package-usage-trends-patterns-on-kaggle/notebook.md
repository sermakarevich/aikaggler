# Python Package Usage Trends & Patterns on Kaggle

- **Author:** Ali Jalaali
- **Votes:** 23
- **Ref:** alijalali4ai/python-package-usage-trends-patterns-on-kaggle
- **URL:** https://www.kaggle.com/code/alijalali4ai/python-package-usage-trends-patterns-on-kaggle
- **Last run:** 2025-07-21 20:18:50.507000

---

<div class="alert alert-block alert-success">
<center><h3 style="color:black;">We have also built an interactive demo that allows users to explore additional plots, filter patterns by their own selected packages, and dynamically visualize trends and associations across the Kaggle ecosystem.</h3> <h1><b><a href="https://ali-jalaali.shinyapps.io/Meta_Kaggle_Hackathon/">Interactive Demo</a></b></h1> </center>
</div>

<h1 style="background: linear-gradient(to right, #FFD700, #FF8C00, #FF4500); color: white; padding: 12px; border-radius: 6px; font-family: Arial; text-align: center; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">0- Imports & Util Functions</h1>

```python
!pip install -q nbformat
!pip install -q scikit-misc
```

```python
import numpy as np
import pandas as pd
import nbformat
import ast
import kagglehub
from kagglehub import KaggleDatasetAdapter
from os import listdir
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')
from IPython.display import display,clear_output,HTML
from plotnine import *
from datetime import datetime, timedelta
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth, association_rules
```

```python
def get_ipynb_packages(notebook_path):
    with open(notebook_path, 'r') as file:
        nb = nbformat.read(file, as_version=4)
    packages = set()
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            try:
                tree = ast.parse(cell['source'])
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            packages.add(name.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        packages.add(node.module.split('.')[0])
            except SyntaxError:
                continue 
    return set(packages)


def analyze_whole_directory(target_dir , limit_n=None):
    df = pd.DataFrame()
    for l1_dir in tqdm(listdir(target_dir)):
        for f in listdir(target_dir+l1_dir):
            try:
                fid = f.replace('.ipynb','')
                df = pd.concat( [df , pd.DataFrame([{'id':fid , 'package':p} for p in get_ipynb_packages(target_dir+l1_dir+'/'+f)])] )
            except: pass
        try:
            if limit_n is not None and df[['id']].nunique()['id'] > limit_n:
                break
        except: pass
    return df


def analyze_single_code(kernel_id):
    dir0 = '/kaggle/input/meta-kaggle-code/'
    kernel_id_z = ('0'*(10-len(kernel_id))) + kernel_id
    dir1 = kernel_id_z[:4]
    dir2 = kernel_id_z[4:7]
    f = kernel_id+'.ipynb'
    # print(dir0+dir1+'/'+dir2+'/'+f)
    return pd.DataFrame([{'id':kernel_id , 'package':p} for p in get_ipynb_packages(dir0+dir1+'/'+dir2+'/'+f)])



def analyze_codes_of_date(K , date ):
    df = pd.DataFrame()
    for kvid in tqdm(K[K['CreationDate'].str.startswith(date, na=False)]['CurrentKernelVersionId']):
        try:
            df = pd.concat([ df , analyze_single_code(str(int(kvid))) ])
        except: pass
    df['date'] = date
    return df

def analyze_codes_in_date_range(K , start_date_str, end_date_str, jump_days=1):
    date_format = "%m/%d/%Y"
    start_date = datetime.strptime(start_date_str, date_format)
    end_date = datetime.strptime(end_date_str, date_format)    
    current_date = start_date
    df = pd.DataFrame()
    while current_date < end_date:
        print(current_date,end='\t')
        df = pd.concat([ df , analyze_codes_of_date(K , current_date.strftime(date_format) )])
        current_date += timedelta(days=jump_days)
    clear_output() 
    print("> Total kernels analyzed:" , df[['id']].nunique()['id'] )
    return df


def count_values_and_summarize(df):
    packages_count = pd.DataFrame(df[['package']].value_counts()).rename_axis('package').reset_index()
    packages_count = packages_count.rename(columns={'count': 'usage_count'})
    packages_count['usage_ratio'] = packages_count['usage_count'] / df[['id']].nunique()['id']
    # packages_count.drop(columns=['count'],inplace=True)
    packages_count['rank'] = packages_count['usage_ratio'].rank(ascending=False,method='min')
    return packages_count[['rank','package','usage_count','usage_ratio']]


def display_df(df , title_text):
    styled_df = df.style.hide(axis='index').set_table_styles([ # Center the table on the page
            {'selector': 'table', 'props': [('margin-left','auto !important'), ('margin-right','auto !important'), ('width','100%'), ('max-width','1200px')]}, # Style column headers with light red background
            {'selector': 'th', 'props': [('background-color', '#ff5757'),('color', 'black'),('font-weight','bold'),('text-align','center'),
                                         ('border','1px solid #ddd'),('padding','8px')]}, # Style table cells
            {'selector': 'td', 'props': [('border','1px solid #ddd'),('padding','8px'),('text-align','center')]}, # Hover effect for rows
            {'selector': 'tr:hover','props': [('font-weight','bold')]}
        ]).set_properties(**{'font-family':'Arial, sans-serif', 'font-size':'14px' # Consistent font styling
        }).format({ 'Sales': '{:,.0f}' # Format Sales column with thousands separator
        }).set_caption( '<div style="text-align: center; font-size: 18px; font-weight: bold; background-color:blue;padding:12px;border-radius:6px;" '
                        'font-family: Arial, sans-serif; margin-bottom: 10px;width:100%;max-width:1200px">'+title_text+'</div>')# Centered title above the table

    display(styled_df)
```

```python
USER = 8065392

K = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS,"kaggle/meta-kaggle","Kernels.csv")
cols = ['CurrentKernelVersionId','AuthorUserId','CreationDate','TotalVotes','TotalComments']
K = K[cols]
# display(K[K['AuthorUserId']==USER])
```

---

<h1 style="background: linear-gradient(to right, #FFD700, #FF8C00, #FF4500); color: white; padding: 12px; border-radius: 6px; font-family: Arial; text-align: center; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">1- Most Popular Packages (Jan–Jul 2025)</h1>

### <center><i>Which Python packages are currently the most popular on Kaggle?</h3></i></center>
<center><i><h3>How widely adopted are most Python packages on Kaggle?</h3></i></center>
<br>

To explore **which Python packages are currently popular on Kaggle**, We analyzed a series of kernels created **between April 1st, 2025 and July 15th**, 2025, **sampling every three days** to ensure temporal coverage while maintaining scalability.

**For each sampled day, We collected all public kernels and extracted the list of imported Python packages.** By aggregating the number of times each package was imported across these kernels, We computed a usage count and a kernel usage ratio for each package — reflecting both raw frequency and relative adoption.

**This experiment aims to surface the most widely used libraries in recent Kaggle activity**, shedding light on current user preferences, popular tools, and potentially emerging trends in the Kaggle data science ecosystem.

```python
df = analyze_codes_in_date_range(K , "01/01/2025", "07/19/2025" , jump_days=10)
packages_counts = count_values_and_summarize(df)
packages_counts.to_csv("packages_counts_2025.csv",index=False)
```

```python
display_df( packages_counts.head(20) , "Most Popular Packages on Kaggle (2025)")
```

This plot shows how Python packages are distributed based on their usage ratio—the proportion of kernels in which each package appears.

The x-axis represents the usage ratio (from rare to widely used packages), while the y-axis shows the number of packages at each usage level.

```python
ggplot(packages_counts) + geom_histogram(aes("usage_ratio"),bins=70,fill='#0000FF') + scale_x_log10() + scale_y_log10() + theme(figure_size=(14,5))+\
labs(title="Distribution of Package Usage Ratios" , subtitle="Highlighting how a small number of packages are heavily used, while most have low adoption rates",
    x="Usage Ratio of Package",y="Number of Packages") +\
theme( plot_title=element_text(hjust=0.5), plot_subtitle=element_text(hjust=0.5) )
```

---

<h1 style="background: linear-gradient(to right, #FFD700, #FF8C00, #FF4500); color: white; padding: 12px; border-radius: 6px; font-family: Arial; text-align: center; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">2- Number of Packages Imported vs Votes of a Kernel</h1>

<center><i><h3>Does using more packages make your kernel more impressive or just more complicated?</h3></i></center>
<center><i><h3>Do Kaggle notebooks with more imported packages get more upvotes or does simplicity win?</h3></i></center>
<center><i><h3>Is there a sweet spot for how many libraries a Kaggle kernel should use?</h3></i></center>
<br>
This analysis explores how the number of imported Python packages in a Kaggle kernel relates to its upvote count, which serves as a proxy for community appreciation.

The x-axis represents the number of distinct packages imported in each kernel, while the y-axis shows the total upvotes received. The goal is to identify whether kernels that use more libraries are generally perceived as more valuable—or whether there is a point of diminishing returns, where complexity may reduce accessibility or clarity.

Patterns in the plot can reveal whether concise, focused notebooks or tool-rich, complex ones tend to be better received by the Kaggle community.

```python
df = analyze_codes_in_date_range(K , "05/01/2025", "05/06/2025"  )
num_packages_used = df.groupby('id')[['package']].count().reset_index()
num_packages_used['id'] = num_packages_used['id'].astype(float).astype(int)
mrg = pd.merge( K , num_packages_used , left_on='CurrentKernelVersionId' , right_on='id')
```

```python
print("> Correlationship between num of packages imported and num of votes:", mrg['package'].corr(mrg['TotalVotes']) )
```

```python
mrg['package'] = mrg['package'].astype(int)
ggplot(mrg) + geom_jitter(aes('package','TotalVotes')) + scale_y_log10() +\
geom_smooth(aes('package','TotalVotes'),method="loess", se=True, color="blue",span=0.5)+ theme(figure_size=(13,5)) +\
labs(x="Number of Packages Imported in kernel", y="Upvotes of Kernel",title="Number of Packages Imported vs Votes of a Kernel")
```

```python
mrg['package'] = pd.Categorical(mrg['package'].astype(str),categories=sorted(mrg['package'].unique().astype(str), key=int),ordered=True)
ggplot(mrg) + geom_boxplot(aes('package','TotalVotes')) + scale_y_log10() + theme(figure_size=(13,5)) +\
labs(x="Number of Packages Imported in kernel", y="Upvotes of Kernel",title="Number of Packages Imported vs Votes of a Kernel")
```

---

<h1 style="background: linear-gradient(to right, #FFD700, #FF8C00, #FF4500); color: white; padding: 12px; border-radius: 6px; font-family: Arial; text-align: center; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">3- Trends in the Number of Imported Packages over the Years</h1>

<center><i><h3>Have Kaggle users started using more packages per notebook over the years?</h3></i></center>
<center><i><h3>Are modern Kaggle notebooks more sophisticated than they were a few years ago?</h3></i></center>
<br>
To investigate whether Kaggle kernels are becoming more complex, We analyzed notebooks sampled from the same calendar days between 2016 and 2025. For each sampled kernel, We counted the number of distinct Python packages imported.

By aggregating this data yearly, we can observe trends in how the average number of imported packages per kernel has changed over time. This provides insight into whether notebooks are becoming more tool-rich, potentially reflecting more advanced modeling workflows, growing library ecosystems, or increased task complexity.

```python
num_packages_used_over_years = pd.DataFrame()

df = analyze_codes_in_date_range(K , "05/01/2016", "07/01/2016" , 20 )
num_packages_used = df.groupby('id')[['package']].count().reset_index()
num_packages_used['year'] = 2016
num_packages_used_over_years = pd.concat([num_packages_used_over_years  , num_packages_used])

df = analyze_codes_in_date_range(K , "05/01/2017", "07/01/2017" , 20 )
num_packages_used = df.groupby('id')[['package']].count().reset_index()
num_packages_used['year'] = 2017
num_packages_used_over_years = pd.concat([num_packages_used_over_years  , num_packages_used])

df = analyze_codes_in_date_range(K , "05/01/2018", "07/01/2018" , 20 )
num_packages_used = df.groupby('id')[['package']].count().reset_index()
num_packages_used['year'] = 2018
num_packages_used_over_years = pd.concat([num_packages_used_over_years  , num_packages_used])

df = analyze_codes_in_date_range(K , "05/01/2019", "07/01/2019" , 20 )
num_packages_used = df.groupby('id')[['package']].count().reset_index()
num_packages_used['year'] = 2019
num_packages_used_over_years = pd.concat([num_packages_used_over_years  , num_packages_used])

df = analyze_codes_in_date_range(K , "05/01/2020", "07/01/2020" , 20 )
num_packages_used = df.groupby('id')[['package']].count().reset_index()
num_packages_used['year'] = 2020
num_packages_used_over_years = pd.concat([num_packages_used_over_years  , num_packages_used])

df = analyze_codes_in_date_range(K , "05/01/2021", "07/01/2021" , 20 )
num_packages_used = df.groupby('id')[['package']].count().reset_index()
num_packages_used['year'] = 2021
num_packages_used_over_years = pd.concat([num_packages_used_over_years  , num_packages_used])

df = analyze_codes_in_date_range(K , "05/01/2022", "07/01/2022" , 20 )
num_packages_used = df.groupby('id')[['package']].count().reset_index()
num_packages_used['year'] = 2022
num_packages_used_over_years = pd.concat([num_packages_used_over_years  , num_packages_used])

df = analyze_codes_in_date_range(K , "05/01/2023", "07/01/2023" , 20 )
num_packages_used = df.groupby('id')[['package']].count().reset_index()
num_packages_used['year'] = 2023
num_packages_used_over_years = pd.concat([num_packages_used_over_years  , num_packages_used])

df = analyze_codes_in_date_range(K , "05/01/2024", "07/01/2024" , 20 )
num_packages_used = df.groupby('id')[['package']].count().reset_index()
num_packages_used['year'] = 2024
num_packages_used_over_years = pd.concat([num_packages_used_over_years  , num_packages_used])

df = analyze_codes_in_date_range(K , "05/01/2025", "07/01/2025" , 20 )
num_packages_used = df.groupby('id')[['package']].count().reset_index()
num_packages_used['year'] = 2025
num_packages_used_over_years = pd.concat([num_packages_used_over_years  , num_packages_used])

num_packages_used_over_years['year'] = num_packages_used_over_years['year'].astype(str)
```

```python
ggplot(num_packages_used_over_years,aes('year','package')) + geom_violin(width=1.1) + geom_boxplot(width=0.15) + scale_y_log10() +\
theme(figure_size=(13,5)) + labs(x="Year",y="Number of Packages Imported in kernel",title="Trends in the Number of Imported Packages over the Years")
```

---

<h1 style="background: linear-gradient(to right, #FFD700, #FF8C00, #FF4500); color: white; padding: 12px; border-radius: 6px; font-family: Arial; text-align: center; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">4- Trending packages</h1>

<center><i><h3>Which packages were already used in 2023 but have gained significant traction in 2025?</h3></i></center>
<center><i><h3>What new packages have emerged since 2023 and become widely used in 2025?</h3></i></center>
<br>
This section identifies trending Python packages by comparing usage between 2023 and 2025. The goal is to highlight tools that are either rising in popularity or newly adopted by the Kaggle community. We categorize trending packages into two groups:


* **Rising Veterans**:
Packages that were already used in 2023, but have seen significant growth in usage by 2025. These tools are gaining momentum and becoming more central to the Kaggle workflow.

* **Newcomers**:
Packages that had little or no usage in 2023, but are now prominently used in 2025. These represent emerging tools or libraries driven by recent trends in machine learning, deep learning, or data engineering.

```python
df = analyze_codes_in_date_range(K , "01/01/2023", "07/19/2023" , jump_days=10)
packages_counts_2023 = count_values_and_summarize(df)
```

```python
df = analyze_codes_in_date_range(K , "01/01/2025", "07/19/2025" , jump_days=10)
packages_counts_2025 = count_values_and_summarize(df)
```

```python
packages_counts_2023.drop(columns='usage_count',inplace=True)
packages_counts_2025.drop(columns='usage_count',inplace=True)

packages_counts_2025 = packages_counts_2025[packages_counts_2025['usage_ratio']>0.001]
```

```python
comparison = pd.merge(packages_counts_2023,packages_counts_2025,on="package",how='right',suffixes=('_2023', '_2025'))
comparison['growth_ratio'] = comparison['usage_ratio_2025'] / comparison['usage_ratio_2023']
comparison = comparison.sort_values(by=['growth_ratio'], ascending=False)
comparison = comparison[["package",'growth_ratio','usage_ratio_2023','usage_ratio_2025','rank_2023','rank_2025']]
comparison.to_csv("2023_2025_comparison.csv",index=False)
```

```python
display_df( comparison.head(20) , "Packages that were already used in 2023, but have seen significant growth in usage by 2025")
```

```python
display_df( comparison[comparison['usage_ratio_2023'].isna()] , "Packages that had no usage in 2023, but are now prominently used in 2025")
```

---

<h1 style="background: linear-gradient(to right, #FFD700, #FF8C00, #FF4500); color: white; padding: 12px; border-radius: 6px; font-family: Arial; text-align: center; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">5- Packages Popularity Trend & Ranking Shifts over Time</h1>

<center><i><h4>Which libraries are rising stars and which are fading?</h4></i></center>
<center><i><h4>How have data scientists’ favorite tools changed over time?</h4></i></center>
<br>
To understand how the Kaggle community’s tool preferences have changed over time, this section analyzes the longitudinal trends in Python package usage. By examining how the ranking and popularity of libraries shift year over year, we gain insights into broader shifts in the data science landscape—from the continued reliance on foundational tools like pandas and matplotlib to the growing adoption of modern frameworks such as transformers.

The following plots highlight key trends, such as the decline of conventional packages in relative usage and the emergence of cutting-edge libraries tailored to deep learning, NLP, and generative AI tasks. These trends reflect how the field evolves with new challenges, models, and technologies.

```python
packages_counts_over_years = pd.DataFrame()

df = analyze_codes_in_date_range(K , "01/01/2016", "07/19/2016" , 10 )
packages_counts = count_values_and_summarize(df)
packages_counts['year'] = 2016
packages_counts_over_years = pd.concat([packages_counts_over_years  , packages_counts])

df = analyze_codes_in_date_range(K , "01/01/2017", "07/19/2017" , 10 )
packages_counts = count_values_and_summarize(df)
packages_counts['year'] = 2017
packages_counts_over_years = pd.concat([packages_counts_over_years  , packages_counts])

df = analyze_codes_in_date_range(K , "01/01/2018", "07/19/2018" , 10 )
packages_counts = count_values_and_summarize(df)
packages_counts['year'] = 2018
packages_counts_over_years = pd.concat([packages_counts_over_years  , packages_counts])

df = analyze_codes_in_date_range(K , "01/01/2019", "07/19/2019" , 10 )
packages_counts = count_values_and_summarize(df)
packages_counts['year'] = 2019
packages_counts_over_years = pd.concat([packages_counts_over_years  , packages_counts])

df = analyze_codes_in_date_range(K , "01/01/2020", "07/19/2020" , 10 )
packages_counts = count_values_and_summarize(df)
packages_counts['year'] = 2020
packages_counts_over_years = pd.concat([packages_counts_over_years  , packages_counts])

df = analyze_codes_in_date_range(K , "01/01/2021", "07/19/2021" , 10 )
packages_counts = count_values_and_summarize(df)
packages_counts['year'] = 2021
packages_counts_over_years = pd.concat([packages_counts_over_years  , packages_counts])

df = analyze_codes_in_date_range(K , "01/01/2022", "07/19/2022" , 10 )
packages_counts = count_values_and_summarize(df)
packages_counts['year'] = 2022
packages_counts_over_years = pd.concat([packages_counts_over_years  , packages_counts])

df = analyze_codes_in_date_range(K , "01/01/2023", "07/19/2023" , 10 )
packages_counts = count_values_and_summarize(df)
packages_counts['year'] = 2023
packages_counts_over_years = pd.concat([packages_counts_over_years  , packages_counts])

df = analyze_codes_in_date_range(K , "01/01/2024", "07/19/2024" , 10 )
packages_counts = count_values_and_summarize(df)
packages_counts['year'] = 2024
packages_counts_over_years = pd.concat([packages_counts_over_years  , packages_counts])

df = analyze_codes_in_date_range(K , "01/01/2025", "07/19/2025" , 10 )
packages_counts = count_values_and_summarize(df)
packages_counts['year'] = 2025
packages_counts_over_years = pd.concat([packages_counts_over_years  , packages_counts])

packages_counts_over_years.to_csv("packages_counts_over_years.csv",index=False)
```

```python
def visualize_trend_over_time(target_packages):
    return ggplot(packages_counts_over_years[packages_counts_over_years['package'].isin(target_packages)]) +\
    geom_line( aes(x ='year', y='rank', color='package') , size=1) + scale_y_reverse() +\
    theme(figure_size=(14,4)) + scale_x_continuous(breaks=[x for x in range(2016,2026)]) +\
    theme( plot_title=element_text(hjust=0.5), plot_subtitle=element_text(hjust=0.5), plot_caption=element_text(hjust=0.5) )
```

```python
visualize_trend_over_time(target_packages=['nltk','spacy','gensim','transformers']) +\
labs(subtitle="Rise of Transformers Amid Declining Traditional NLP Libraries",
     caption="While classic NLP libraries like nltk, spacy, and gensim show a gradual decline in usage, transformers has seen a sharp rise—reflecting the shift toward deep learning–based language models.")
```

```python
visualize_trend_over_time(target_packages=['keras','tensorflow','torch']) +\
labs(subtitle="PyTorch Surges Ahead as Keras Declines",
     caption="While keras has experienced a significant decline, torch (PyTorch) has seen rapid growth in adoption,\n reflecting the community’s shift toward more flexible and research-friendly deep learning frameworks")
```

<div class="alert alert-block alert-success">
<center><h3 style="color:black;">We have also built an interactive demo that allows users to explore additional plots, filter patterns by their own selected packages, and dynamically visualize trends and associations across the Kaggle ecosystem.</h3> <h1><b><a href="https://ali-jalaali.shinyapps.io/Meta_Kaggle_Hackathon/">Interactive Demo</a></b></h1> </center>
</div>

# 

<h1 style="background: linear-gradient(to right, #FFD700, #FF8C00, #FF4500); color: white; padding: 12px; border-radius: 6px; font-family: Arial; text-align: center; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">6- Exploring Co‑import Patterns with Market Basket Analysis</h1>

<center><i><h4>What packages tend to appear together in Kaggle notebooks?</h4></i></center>
<center><i><h4>What do these combinations reveal about user workflows?</h4></i></center>
<br>
In this section, I apply Market Basket Analysis—a method commonly used in retail analytics—to Kaggle kernels. Each kernel is treated like a "basket" of imported packages. Using the Apriori algorithm, I identify:

* Frequent itemsets: Groups of packages that are often used together.
* Association rules: If a kernel uses package A, how likely is it to also use package B?

These patterns help us uncover common package pairings or stacks (e.g., pandas + seaborn, or torch + torchvision), and may reflect typical workflows in data analysis, machine learning, computer vision, and NLP. This analysis provides insight into how Kaggle users combine tools—and can even inform recommendations for complementary packages or starter environments.

```python
df = analyze_codes_in_date_range(K , "01/01/2025", "07/19/2025" , jump_days=10)
dataset = []
for kid in tqdm(df['id'].unique()):
    dataset.append( [x for x in df[df['id']==kid]['package']] )
```

```python
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
occurrence_df = pd.DataFrame(te_ary, columns=te.columns_)
occurrence_df.sample(3)
```

```python
# frequent_itemsets = fpgrowth(occurrence_df, min_support=0.01, use_colnames=True)
frequent_itemsets = apriori(occurrence_df, min_support=0.005, use_colnames=True)
# frequent_itemsets = fpmax(occurrence_df, min_support=0.6, use_colnames=True)
print("> Number of frequent itemsets extracted: ",frequent_itemsets.shape[0])
frequent_itemsets = frequent_itemsets.sort_values(["support"],ascending = False)
frequent_itemsets.to_csv("frequent_itemsets.csv",index=False)
frequent_itemsets_display = frequent_itemsets.copy()
frequent_itemsets_display['itemsets'] = [" , ".join(list(i)) for i in frequent_itemsets_display['itemsets']]
```

```python
display_df(frequent_itemsets_display[[len(i)==2 for i in frequent_itemsets['itemsets']]].head(10) , 'Top 10 Frequent Pair of Packages ')
```

```python
display_df( frequent_itemsets_display[[len(i)>3 for i in frequent_itemsets['itemsets']]].head(10) , "Top 10 Frequent big package sets!" )
```

```python
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=3)
print("> Number of rules extracted: ", rules.shape[0] )
rules = rules.sort_values(["lift"],axis = 0, ascending = False)
```

```python
rules.drop(columns=['antecedent support','consequent support','representativity','leverage','conviction','zhangs_metric','jaccard','certainty','kulczynski'],
          inplace=True)
rules.to_csv("rules.csv",index=False)

rules_display = rules.copy()
rules_display['antecedents'] = [" , ".join(list(i)) for i in rules_display['antecedents']]
rules_display['consequents'] = [" , ".join(list(i)) for i in rules_display['consequents']]
display_df(rules_display.head(14) , "Top Association rules")
```

<div class="alert alert-block alert-success">
<center><h3 style="color:black;">We have also built an interactive demo that allows users to explore additional plots, filter patterns by their own selected packages, and dynamically visualize trends and associations across the Kaggle ecosystem.</h3> <h1><b><a href="https://ali-jalaali.shinyapps.io/Meta_Kaggle_Hackathon/">Interactive Demo</a></b></h1> </center>
</div>