# Brazilian Archaeological Radiocarbon dating

- **Author:** Marília Prata
- **Votes:** 57
- **Ref:** mpwolke/brazilian-archaeological-radiocarbon-dating
- **URL:** https://www.kaggle.com/code/mpwolke/brazilian-archaeological-radiocarbon-dating
- **Last run:** 2025-05-20 11:55:28.820000

---

Published on May 19, 2025. By Prata, Marília (mpwolke)

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go

import plotly
plotly.offline.init_notebook_mode(connected=True)

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

## Competition Citation

@misc{openai-to-z-challenge,

author = {Chris Fisher and Philip Bogdanov and Paul Mooney and Nate Keating and Maggie Demkin},

title = {OpenAI to Z Challenge},

year = {2025},

howpublished = {\url{https://kaggle.com/competitions/openai-to-z-challenge}},

## Radiocarbon dating from Brazilian bioarchaeological samples

Stable and radiogenic isotope data from Brazilian bioarchaeological samples: a synthesis

Dataset Citation: Borges, C.,Chanca, I, & Salesse, K. (2021). Stable and radiogenic isotope data from Brazilian bioarchaeological samples: a synthesis. IsoArcH. https://doi.org/10.48530/isoarch.2021.005

"Three decades have passed since the first publication, in 1991, of stable isotope analysis in a Brazilian archaeological context. These data were used in a palaeodietary study and it is presented in the book entitled “Moundbuilders of the Amazon: Geophysical Archaeology on Marajo Island, Brazil” by A.C. Roosevelt. Although still mainly applied to palaeodietary research, stable isotope analysis in archaeology has been diversified in Brazil."

"In the last five years, an increasing number of studies has addressed various research questions, such as population mobility, social differentiation, health and children care, changes and resilience of cultural practices, identification of the origin of enslaved populations brought by force from the African continent, among others."

"However, research in this area is still incipient when compared to the large territory of the country (05˚ to -33˚N, -60˚ to -53˚E), the diversity of socio-cultural contexts of pre-colonial and indigenous societies, and to the country's historical formation process. Thus, the purpose of this compilation was to gather all dispersed, and often fragmented, data from analyses of stable and radioactive (focussing on radiocarbon) isotopes carried out in Brazilian archaeological contexts."

"The authors compiled data published from 1991 until the end of November 2021. The data included here contain information from 71 archaeological sites, 556 humans, 219 animals and 2 plants. Isotopic analyses were performed on 832 organic samples, mainly paired δ13C and δ15N plus δ34S measurements, and on 265 mineral samples, mainly δ13C, δ18O and 86Sr/87Sr measurements. Sr concentrations for 49 mineral samples were also compiled when reported. Samples have radiocarbon or relative dates showing a chronology between 18 kyr BP and nowadays."

"All data from this compilation are deposited in open-access on the IsoArcH platform (https://doi.isoarch.eu/doi/2021.005).  Most of the data compiled here has not been released in peer-reviewed publications, as they comprise master dissertations, doctoral theses and research reports. A number of works included here are published only in local journals, in Portuguese, and comprise peer-reviewed and non-peer-reviewed scientific articles."

"It is important to emphasize that most of the data compiled here might also be of difficult access for Brazilian researchers or Portuguese speakers. This extensive dataset aims to point up the gaps in stable isotopes and radiocarbon estimations provided for Brazilian archaeological contexts that could be further explored and provides easy access to numerous analyses that, otherwise, would be hard to obtain. Lastly, this compilation seeks to broaden interdisciplinary collaboration in Brazil and strengthen the international collaboration among scholars."

https://doi.isoarch.eu/doi/2021.005

## Brazilian radiocarbon bioarchaeological sites

Stable isotope data and radiocarbon dates from Brazilian bioarchaeological samples: An extensive compilation

![](https://ars.els-cdn.com/content/image/1-s2.0-S2352340922003274-gr1.jpg)

## Load Brazilian radiocarbon bioarchaeological samples file

```python
df = pd.read_excel('/kaggle/input/archaeological-sites-map/Brazilian radiocarbon bioarchaeological samples.xlsx')
df.tail()
```

## This challenge suggests to that findings should be bound by the Amazon in Northern South America.

For the record, this dataset focused on the whole Brazil. Besides, the Amazon region is represented here only by the States of Amapá and Pará. And these, didn't provide many samples to be investigated for this Z Challenge (OpenAI).

```python
#StackOverflow https://stackoverflow.com/questions/34076177/matplotlib-horizontal-bar-chart-barh-is-upside-down

df["Region"].value_counts().plot.barh(color=['blue', '#f5005a'], title='Brazilian Archaeological Regions')
plt.gca().invert_yaxis();
```

### The Map below was usefull to determine the coordinates location for the 2nd map.

If I followed the information below to stablish Brazilian location I would get Africa map on the folium map.  

"The Federative Republic of Brazil is located on the geographic coordinates of 10.6500° S latitude and 52.9500° W longitude in South America."

"The main latitude and longitude of Brazil is 10° South and 55° West. The major part of Brazil, which is a South American country, falls in the southern hemisphere.
The total area of Brazil which falls within the first and last latitude and longitude of Brazil is 8,511,965 square kilometer. The latitude and longitude of the capital of Brazil, Brasilia is 15° 45'South and 47°57'West."

https://www.mapsofworld.com/lat_long/brazil-lat-long.html

```python
#By Marília Prata on Kaggle https://www.kaggle.com/code/mpwolke/airports-maps
#StackOverFlow https://stackoverflow.com/questions/25328003/how-can-i-change-the-font-size-using-seaborn-facetgrid

sns.set(font_scale=3) 

plt.figure(figsize=(20,12))
ax = plt.gca()
ax.set_title("Brazilian Archaeological Regions")

g = sns.scatterplot(x='Latitude (wgs 84)', y='Longitude (wgs 84)', data=df, hue='Type of Coordinates')
g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1);
```

## Install Folium

```python
#installation
!pip install folium
```

### Let's try a folium map with more details

```python
#https://medium.com/data-science/using-python-to-create-a-world-map-from-a-list-of-country-names-cd7480d03b10

# Create a world map to show distributions of users 
import folium
from folium.plugins import MarkerCluster
#empty map
bra_map= folium.Map(location=(-33, -60),tiles="cartodbpositron", zoom_start= 3)
marker_cluster = MarkerCluster().add_to(bra_map)
#for each coordinate, create circlemarker of altitude (m)
for i in range(len(df)):
        lat = df.iloc[i]['Latitude (wgs 84)']
        long = df.iloc[i]['Longitude (wgs 84)']
        radius=5
        popup_text = """Region : {}<br>
                    Altitude (m) : {}<br>"""
        popup_text = popup_text.format(df.iloc[i]['Region'],
                                   df.iloc[i]['Altitude (m)']
                                   )
        folium.CircleMarker(location = [lat, long], radius=radius, popup= popup_text, fill =True).add_to(marker_cluster)
#show the map
bra_map
```

## Brazilian bioarchaeological samples info()

```python
df.info()
```

## Load Individuals Radiocarbon dating file

```python
radiocarbon = pd.read_excel('/kaggle/input/archaeological-sites-map/individuals.xlsx')
radiocarbon.tail()
```

## Radiocarbon dating:

A conventional age which has been calibrated to correspond with our solar calendar, stated as calibrated years before present (**cal BP), cal BC/AD, or or cal BCE/CE**.

Source: Radiocarbon Dating Understood
https://www.texasbeyondhistory.net/radiocarbon/dating101.html#:~:text=Radiocarbon%20date%3A%20a%20conventional%20age,or%20or%20cal%20BCE%2FCE.

"**C.E. (Current Era) and B.C.E. (Before Current Era)** are sometimes used instead of A.D. and B.C. as a way to express a date without specifically referencing Christianity. But the dates themselves are exactly comparable to dates expressed as A.D. or B.C. For example, 1280 C.E. is the same year as A.D."

https://crowcanyon.org/education/learn-about-archaeology/archaeological-dating/#:~:text=C.E.%20(Current%20Era)%20and%20B.C.E.,the%20same%20year%20as%20A.D.

A **standard deviation (or σ Sigma)** is a measure of how dispersed the data is in relation to the mean. Low, or small, standard deviation indicates data are clustered tightly around the mean, and high, or large, standard deviation indicates data are more spread out.

https://www.nlm.nih.gov/oet/ed/stats/02-900.html

```python
# Sort data by 14C 2σ Calibrated Date - Lower Limit in Descending order 
radiocarbon_reset = radiocarbon.reset_index()
radiocarbon_reset[["Relative Age - Lower Limit", "Age System", "14C SD (±σ)", "14C 2σ Calibrated Date - Lower Limit"]].sort_values(by = '14C 2σ Calibrated Date - Lower Limit', ascending=False).head(10)
```

## Load individuals funerary file

```python
funerals = pd.read_excel('/kaggle/input/archaeological-sites-map/human_individuals_funerary.xlsx')
funerals.head()
```

### Body disposal type on burials

Inhumation prevailed.

```python
ax = funerals['Disposal Type'].value_counts().plot.barh(figsize=(8, 4), color='green')
ax.set_title('Body Disposal Type', size=18, color='orange')
ax.set_ylabel('Disposal Type', size=10)
ax.set_xlabel('Count', size=10)
plt.gca().invert_yaxis();
```

## Load skeletal part file

```python
skeletal = pd.read_excel('/kaggle/input/archaeological-sites-map/sampled_skeletal_part.xlsx')
skeletal.head()
```

### Skeletal Sample type 

Bone and teeth on top since the majority are human samples.

```python
#Code by Lucas Abrahão https://www.kaggle.com/lucasabrahao/trabalho-manufatura-an-lise-de-dados-no-brasil

sample_proportion = skeletal['Sample Type'].value_counts()/skeletal['Sample Type'].value_counts().sum()
colormap = plt.cm.tab10(range(0, len(sample_proportion)))

#Use logx=True cause the bar didn't appear due to small values

skeletal["Sample Type"].value_counts().plot.barh(logx=True, color=colormap, title='Sample Type')
plt.gca().invert_yaxis();
```

### Sampled Skeletal Part

Ribs and skulls on top of the samples found.

```python
sample_proportion = skeletal['Sampled Skeletal Part'].value_counts()/skeletal['Sampled Skeletal Part'].value_counts().sum()
colormap = plt.cm.tab10(range(0, len(sample_proportion)))

#Use logx=True cause the bar didn't appear due to small values

skeletal["Sampled Skeletal Part"].value_counts().plot.barh(figsize=(18, 12),color=colormap, title='Sampled Skeletal Part')
plt.gca().invert_yaxis();
```

### The 3 types of archaeology

"There are three main branches of study: prehistoric archaeology (**cultures that do not have writing**);"

"Protohistoric archaeology (**cultures that have incomplete records**);"

"And historic archaeology (**cultures that have well-developed historical records**). As such, these branches of archaeology value different types of data.

Additionally, it's better to deliver our records. At least we'll leave our own version from our history.

In other words, publish and share with the community what we have learned. **With or Without AI**.

Also remember that knowledge evolves. So do language/grammar/context. Things will be seen in a different way. That changes the interpretation and conclusions of the material. Sometimes it could seem to be a good work, but analyses are limited by the knowledge that was available at the time.

To determine on the future which parts will still be okay, and which parts are "useless" , it's simply unknown.

That being said, maybe on next time OpenAI.