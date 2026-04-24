# Tupac's California Love Benchmark

- **Author:** Marília Prata
- **Votes:** 34
- **Ref:** mpwolke/tupac-s-california-love-benchmark
- **URL:** https://www.kaggle.com/code/mpwolke/tupac-s-california-love-benchmark
- **Last run:** 2026-03-22 23:37:00.220000

---

Published on March 22, 2026. By Prata, Marília (mpwolke)

## California Love - 2Pac, Dr.Dre, Eazy-E, Ice Cube, Snoop Dogg 

<iframe width="566" height="428" src="https://www.youtube.com/embed/4QkDvurUTWU?list=RD4QkDvurUTWU" title="Dr. Dre - California Love ft. 2Pac, Eazy-E, Ice Cube, Snoop Dogg (Remix)" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

https://www.youtube.com/watch?v=4QkDvurUTWU&list=RD4QkDvurUTWU&start_radio=1

```python
import kaggle_benchmarks as kbench

# STEP 1: DEFINE YOUR TASK

@kbench.task(name="Who was Tupac?", description="Does the LLM know who Tupac was?")
def what_is_kaggle(llm) -> None:

    # A. Prompt the model
    response: str = llm.prompt("Who was Tupac Shakur?")

    # B. Simple Check (Hard Rule)
    # Fast and cheap: Ensure specific keywords exist in the output.
    kbench.assertions.assert_in("rapper", response.lower())

    # C. Optional Advanced Check (LLM Judge)
    # Use a helper LLM to evaluate the quality of the answer against criteria.
    assessment = kbench.assertions.assess_response_with_judge(
        response_text=response,
        judge_llm=kbench.judge_llm,
        criteria=[
            "The answer must mention Las Vegas.",
            "The answer should mention 2Pacalypse Now",
            "The answer should mention hip-hop.",
        ]
    )

    # Iterate through the judge's feedback and assert success
    for result in assessment.results:
        kbench.assertions.assert_true(
            result.passed,
            expectation=f"Judge Criterion '{result.criterion}' should pass: {result.reason}"
        )

# --------------------------------------------------------------------------------
# STEP 2: Generate TASK (avoid quick save it will fail)
# We use `kbench.llm` as a placeholder. 

what_is_kaggle.run(kbench.llm)
```

## google/gemini-2.5-flash Answered:

Tupac Amaru Shakur, known mononymously as 2Pac, was an American rapper, songwriter, actor, and poet who became one of the most iconic and influential figures in hip-hop history. Born Lesane Parish Crooks on June 16, 1971, in East Harlem, New York City, he was later renamed Tupac Amaru Shakur by his mother, Afeni Shakur, a prominent member of the Black Panther Party.

Here's a breakdown of who Tupac Shakur was:

Early Life and Influences:

Black Panther Roots: His upbringing was deeply influenced by the political activism of his mother and stepfather, Mutulu Shakur. This instilled in him a strong sense of social justice, awareness of systemic oppression, and a critical view of society.
Artistic Education: He attended the Baltimore School for the Arts, where he studied acting, poetry, jazz, and ballet. This diverse artistic background contributed to his lyrical depth and theatrical stage presence.
Moving West: He moved to Marin City, California, in the late 1980s, where he began his musical journey.
Musical Career (1990s):

Digital Underground: He began his career as a roadie and backup dancer for the hip-hop group Digital Underground, eventually making his recording debut on their 1991 album Sons of the P.
Solo Breakthrough: His solo debut album, 2Pacalypse Now (1991), was controversial due to its raw depictions of poverty, police brutality, and gang violence, but it immediately established him as a powerful new voice.
Commercial Success: Albums like Strictly 4 My N.I.G.G.A.Z. (1993), Me Against the World (1995), and especially All Eyez on Me (1996) solidified his status as a superstar. All Eyez on Me became one of the best-selling hip-hop albums of all time.
Themes & Style: Tupac's music was characterized by its raw emotion, vivid storytelling, and complex duality. He could be aggressive and confrontational, yet also vulnerable, poetic, and socially conscious. His lyrics often explored themes of racism, poverty, police brutality, the struggles of inner-city life, hedonism, loyalty, and mortality.
Posthumous Releases: After his death, numerous albums and singles were released, continuing his legacy and further demonstrating the depth of his unreleased material.
Acting Career:

Tupac also made a significant mark in Hollywood. His most notable film roles include:
Juice (1992): His breakthrough role as Bishop, which earned him critical acclaim.
Poetic Justice (1993): Starring alongside Janet Jackson.
Above the Rim (1994): A popular basketball-themed drama.
Controversies and Legal Troubles:

Tupac's life was marked by frequent legal battles, arrests, and highly publicized controversies, including charges of sexual assault (for which he was convicted) and various other violent incidents.
His incarceration in 1995 deeply impacted his outlook and music, leading to the creation of some of his most introspective work (Me Against the World was released while he was in prison).
East Coast–West Coast Hip-Hop Rivalry:

Upon his release from prison in 1995, Tupac signed with Suge Knight's Death Row Records. This period saw him become a central figure in the escalating East Coast–West Coast hip-hop rivalry, particularly with Sean "Puffy" Combs and The Notorious B.I.G. from Bad Boy Records.
His diss track "Hit 'Em Up" is considered one of the most aggressive and personal in hip-hop history.
Tragic Death:

On September 7, 1996, Tupac was shot in a drive-by shooting in Las Vegas, Nevada. He succumbed to his injuries six days later, on September 13, 1996, at the age of 25. His murder remains officially unsolved.
His final album, The Don Killuminati: The 7 Day Theory, released under the pseudonym Makaveli, was put out posthumously in November 1996 and was eerily prophetic in its dark themes.
Legacy and Impact:
Tupac Shakur remains a towering and enigmatic figure in popular culture. He is celebrated for his lyrical genius, passionate delivery, complex persona, and his ability to articulate the struggles and aspirations of a generation. His music continues to resonate globally, influencing countless artists across genres. He is often remembered not just as a rapper, but as a poet, an activist, and a symbol of rebellion and resilience.

## Working on a Benchmark Notebook requires installations.

Therefore, I won't go further. Installing torch, stopped the Notebook.

```python
!pip install matplotlib
```

```python
!pip install seaborn
```

```python
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')
```

## Load csv

```python
df = pd.read_csv('/kaggle/input/datasets/leopoldofacci/tupac-dataset-lyrics-analyse/lyrics-TUPAC.csv')

df.head(2)
```

## New column names (Rename columns)

```python
# List of new column names
new_column_names = ['artist', 'title', 'lyrics']

# Assign the new names to the DataFrame
df.columns = new_column_names

df.head(3)
```

```python
!pip install wordcloud
```

```python
#By SRK https://www.kaggle.com/code/sudalairajkumar/simple-exploration-notebook-qiqc

from wordcloud import WordCloud, STOPWORDS

# Thanks : https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##
def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='white',
                    color_func=lambda *args, **kwargs: "red",
                    #font_path=font_file,      
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    width=800, 
                    height=400,
                    mask = mask)
    wordcloud.generate(str(text))
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'black', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  
    
plot_wordcloud(df["artist"], title="Tupac")
```