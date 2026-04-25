# Ötzi, the Iceman. Gemini3 Pro

- **Author:** Marília Prata
- **Votes:** 72
- **Ref:** mpwolke/tzi-the-iceman-gemini3-pro
- **URL:** https://www.kaggle.com/code/mpwolke/tzi-the-iceman-gemini3-pro
- **Last run:** 2025-12-06 14:01:55.223000

---

Published on December 05, 2024. By Prata, Marília (mpwolke)

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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

@misc{gemini-3,

    author = {Amit Vadi and Omar Sanseviero and Paige Bailey and Seth Odoom and Joana Carrasqueira and Ashley Oldacre and Addison Howard},
    
    title = {Google DeepMind - Vibe Code with Gemini 3 Pro in AI Studio },
    year = {2025},
    
    howpublished = {\url{https://kaggle.com/competitions/gemini-3}},
    note = {Kaggle}
}

## Why Ötzi the Iceman is so important and what's his curse?

I opened Google Gemini Studio Gallery page and asked the question above: https://aistudio.google.com/apps?source=

That's what I proposed to Gemini. (Why Ötzi the Iceman is so important and what's his curse?)

Thought opened in 22 seconds

https://aistudio.google.com/apps/temp/1?source=&showAssistant=true&showPreview=true

![image.png](attachment:5f4e5520-2697-4519-8931-032c698bc2eb.png)

## The Blue button "Why is he important?"

The 1st page was Discovery. After clicking on the blue button, it changed to Significance section. Where the text below appears.


https://aistudio.google.com/apps/temp/1?source=&showAssistant=true&showPreview=true

"Ötzi the Iceman, a naturally mummified man discovered in the Ötztal Alps in 1991, is an unparalleled scientific treasure, offering a direct window into life during the Copper Age in Europe around 5,300 years ago. His remarkable preservation, along with his clothing and equipment, has allowed scientists to glean extensive and detailed insights into ancient human biology, health, technology, diet, and even social structures.

Preservation
Ötzi's extraordinary preservation is due to a rare natural phenomenon known as cryopreservation. After his death, his body was quickly covered by snow and ice. The cold, dry, and oxygen-deprived conditions of the glacier prevented decomposition by bacteria and microorganisms, effectively "flash-freezing" him. This natural mummification process preserved not only his skeleton but also his skin, muscles, organs, blood, hair, clothing, and even the contents of his stomach in remarkable detail. Unlike most mummies, which are often treated with substances and have organs removed as part of ritualistic burial, Ötzi was preserved almost entirely intact. He is currently housed in a custom cold chamber at the South Tyrol Museum of Archaeology in Bolzano, Italy, maintained at a constant temperature of -6 degrees Celsius (21.2 degrees Fahrenheit) and 99% humidity, and is regularly sprayed with sterile water to maintain his "wet mummy" state.

Tattoos
Ötzi's body is adorned with 61 tattoos, consisting of groups of black lines and crosses found on his lower back, abdomen, left wrist, lower legs, and behind his right knee. These tattoos, which are 1-3 millimeters wide and 7-40 millimeters long, were made from pigment derived from fireplace ash or soot. Recent research challenges the long-held belief that they were created by rubbing soot into incisions. Instead, a 2024 study suggests they were likely made using a "hand-poking" method with a single-pointed tool, possibly made of bone or copper, to punch closely spaced holes in the skin. The purpose of these tattoos is a subject of ongoing speculation; while some researchers have suggested they might have had medicinal purposes, similar to early acupuncture, due to their location on areas corresponding to modern acupuncture points and showing signs of wear and tear, their exact significance remains unclear.

Medical Condition
Detailed scientific analyses have revealed a comprehensive picture of Ötzi's health, offering insights into ancient diseases and medical practices:

Predisposition to Diseases: DNA analysis revealed that Ötzi was predisposed to cardiovascular disease, including arteriosclerosis (hardened arteries), despite his active lifestyle. He also had brown eyes, blood type O, and was lactose intolerant, a common trait in Neolithic agrarian societies.
Infections and Parasites: Ötzi was the earliest known human with Lyme disease, as indicated by the presence of Borrelia burgdorferi DNA in his hip bone tissue. He also suffered from intestinal parasites, specifically whipworm (Trichuris trichiura), and had Helicobacter pylori, a bacterium that causes stomach ulcers and gastritis, though the strain found is uncommon in modern Europeans.
Injuries and Trauma: He had considerable tooth decay, gum disease, and worn teeth, as well as arthritis in his hips, knees, ankles, and spine. One of his fingernails showed three Beau's lines, indicating he was sick three times in the six months before his death, with the last illness lasting about two weeks, two months before he died. His lungs were coated with soot, suggesting frequent exposure to open fires. Ötzi's death was violent; he had an arrowhead lodged in his left shoulder, which likely caused fatal blood loss, and also sustained defensive wounds to his hands and arms, as well as cerebral trauma from a blow to the head.
Stomach Contents
Analysis of Ötzi's stomach contents has provided invaluable information about his last meals and the Copper Age diet. His stomach, which had shifted upwards during mummification, was remarkably well-preserved. Researchers identified the remains of a fatty, hearty meal consumed less than two hours before his death. This meal included fat and meat from ibex and red deer, as well as einkorn wheat. The high proportion of animal fat (46% of his stomach contents) suggests he ate a meal adjusted to the energetic requirements of high-altitude trekking. Microscopic and spectroscopic analyses indicate he consumed fresh or dried wild meat, possibly smoked. The presence of charcoal particles in his intestines further supports the idea of food preparation using fire, likely for drying or smoking meat. Pollen grains of hop-hornbeam found in his stomach suggest his death occurred in the spring or early summer.

Copper Axe
Ötzi's copper axe is one of his most significant artifacts. It features a 60 cm long yew wood haft with a 9.5 cm long blade made of almost pure copper (99.7%). This axe is considered the world's oldest completely preserved axe from the Neolithic period, including its handle, leather bindings, and birch tar adhesive. The axe's importance lies in its demonstration of sophisticated metalworking techniques previously underestimated for this era. Isotopic analysis revealed that the copper in the axe blade did not come from the Alps, but rather from southern Tuscany, approximately 500 kilometers away. This finding provides concrete evidence of extensive long-distance trade networks and cultural exchange operating in Copper Age Europe, connecting central Italy with the Alpine regions.

Insights into the Copper Age in Europe
Ötzi offers an unparalleled snapshot of Copper Age life in Europe (3359 to 3105 BCE).

Technological Advancement: His tools and equipment, including the sophisticated copper axe, flint dagger, bow, arrows, and quiver, reveal the advanced craftsmanship and technological capabilities of the time. The copper axe, in particular, highlights early metallurgical knowledge and resourcefulness.
Diet and Lifestyle: His stomach contents confirm a diet that included both wild game (ibex, red deer) and cultivated cereals (einkorn wheat), indicating a lifestyle of both hunting and farming. The high fat content of his last meal suggests a pragmatic approach to energy intake for arduous mountain journeys.
Trade and Connections: The Tuscan origin of his copper axe provides direct evidence of long-distance trade routes across challenging mountain terrain, demonstrating significant inter-regional connections and cultural exchange previously not well understood. This implies complex social structures and organized commerce.
Health and Disease: Ötzi's medical conditions offer a unique look at the health challenges faced by people in the Copper Age, including the prevalence of diseases like Lyme disease and gastrointestinal issues, as well as degenerative conditions.
Social Context and Violence: The fatal arrow wound and defensive injuries suggest that violence and conflict were part of life in the Copper Age, possibly related to raiding or personal disputes. His presence at such a high altitude, where people did not typically live year-round, may indicate he was fleeing or engaged in conflict.
Ancestry: DNA analysis shows Ötzi traced the majority of his ancestry to Neolithic early European farmers who migrated from Anatolia, replacing earlier hunter-gatherer populations. More recent studies confirm a high percentage of Anatolian farmer ancestry and suggest he had dark skin and a genetic predisposition for adult-onset diabetes, obesity, and male pattern baldness.
In essence, Ötzi is not merely an archaeological find; he is a meticulously preserved individual whose body, belongings, and circumstances of death have collectively transformed our understanding of a pivotal period in European prehistory."

https://aistudio.google.com/apps/temp/1?source=&showAssistant=true&showPreview=true

## Sources and Further Reading

Sources and Further Reading were provided there:

https://aistudio.google.com/apps/temp/1?source=&showAssistant=true&showPreview=true

![image.png](attachment:d26a64d0-6eb4-4daf-a063-3de2893b4266.png)

## The Curse of Ötzi

Then, I clicked on the button "The Curse". That was the text.

Deep in the Ötztal Alps, where the ice grips the mountains with an ancient embrace, lies the saga of Ötzi the Iceman – a 5,300-year-old Copper Age warrior whose remarkably preserved body emerged from its glacial tomb in 1991. Yet, for all the scientific wonder he inspired, Ötzi also brought with him a chilling narrative: the whispered "Curse of Ötzi," a series of untimely deaths among those intimately connected to his discovery and research. Was it merely a string of tragic coincidences, or did the Iceman, disturbed from his millennia-long slumber, exact a silent revenge?

The tale of the curse began to unfurl shortly after Ötzi's unearthing, casting a shadow over the scientific marvel. The alleged victims, often prominent figures in the Ötzi narrative, met their ends in various unfortunate ways, prompting speculation about a supernatural influence.

First among those linked to the "curse" was Rainer Henn, the forensic pathologist from Innsbruck University who was among the first to examine Ötzi's body, even reportedly handling the cadaver with his bare hands. In 1992, just a year after Ötzi's discovery, Henn, aged 64, tragically died in a car crash while en route to a conference where he was scheduled to present his findings on the Iceman.

Next came Kurt Fritz, a seasoned mountaineer who had guided Rainer Henn to Ötzi's remote resting place. In 1993, at 52 years old, Fritz was reportedly the sole victim of an avalanche during a climbing expedition, an experienced climber lost in familiar terrain.

The chilling pattern continued with Rainer Hoelzl, a prominent Austrian journalist who had exclusive access to film Ötzi's recovery. Months after releasing an hour-long documentary about the excavation, Hoelzl died at the age of 47 from a brain tumor.

Perhaps the most poignant and eerie death was that of Helmut Simon, the German tourist who, along with his wife, originally discovered Ötzi in 1991. In October 2004, thirteen years after his momentous discovery, Simon, aged 67, disappeared during a solo hike in the very same Alpine region. His body was found eight days later, frozen in ice near the spot where he had found Ötzi, having fallen to his death during a sudden blizzard.

The grim sequence didn't end there. Within an hour of Helmut Simon's funeral, Dieter Warnecke, the head of the mountain rescue team tasked with finding Simon, suffered a fatal heart attack at the age of 45.

Then, in 2005, the highly respected archaeologist Konrad Spindler passed away at 66 from complications arising from multiple sclerosis. Spindler was the first archaeologist to officially examine Ötzi and had famously scoffed at the idea of a curse, reportedly declaring, "It is all a media hype. The next thing you will be saying I will be next."

Finally, the alleged curse claimed Dr. Tom Loy, a molecular archaeologist from the University of Queensland, Australia. Loy conducted groundbreaking DNA analysis on Ötzi, which revealed traces of human blood from multiple individuals on the Iceman's clothes and weapons, suggesting a violent confrontation leading to his death. Loy, who did not believe in the curse and took a scientific approach to the pattern, died in 2005 at the age of 63 in his Brisbane home, on the cusp of completing a book about his research. He had suffered from a blood-related condition for about 12 years, diagnosed shortly after his involvement with the Iceman project.

Statistical Coincidence or Something More?

The narrative of the "Curse of Ötzi" is undeniably compelling, drawing parallels to the famous "Curse of Tutankhamun." The series of unexpected deaths, particularly those involving individuals in seemingly good health or succumbing to unusual circumstances, fuels the mysterious allure. It's easy to connect these dots and imagine an ancient spirit angered by disturbance.

However, from a scientific and statistical perspective, these events are widely considered to be coincidences. Hundreds of scientists, researchers, journalists, and support personnel have been involved in the study and display of Ötzi over the decades. In any large group of people over an extended period, it is inevitable that some will experience illness, accidents, or premature deaths. The human tendency to seek patterns, even in randomness, can lead to the creation of such "curses."

For example, Konrad Spindler died from complications of multiple sclerosis, a degenerative disease, at an age that, while not exceptionally old, is also not exceptionally young. Rainer Henn's car crash, while tragic, is a common cause of death. Helmut Simon's death in the Alps, while a cruel twist of fate given his discovery, occurred in a high-risk environment for even experienced mountaineers, and his body was found frozen in similar conditions to Ötzi's, not as a result of supernatural intervention. Tom Loy's death, while untimely, was linked to a pre-existing blood condition.

The alleged "curse" captivates our imagination, adding a layer of mystique to an already extraordinary archaeological find. But for the world of science, these unfortunate events, while sad, are generally attributed to the natural probabilities of life and the inherent risks associated with exploration and old age, rather than the wrath of a Copper Age iceman. The legend, however, continues to be a captivating footnote in the ongoing story of Ötzi.

https://aistudio.google.com/apps/temp/1?source=&showAssistant=true&showPreview=true

## Ask Expert button

Since I respect Ötzi's Curse, I won't ask for anything else that Gemini3 Pro Preview could Not answer. Now, it's up to you.

![image.png](attachment:3cffcd60-270b-4d3d-8368-c928ea1481fb.png)

## Opening My md text file

That's the text file used to make my dataset. 

### My text was ignored by Gemini3 Pro.

```python
# http://www.gutenberg.org/cache/epub/730/pg730.txt
text = open("../input/tzi-iceman/tzi The Iceman.md", "r", encoding ='cp1251').read()
print (text[:4000])
```

## I'm the Keeper of the Iceman's story. 

## Tell that to Ötzi Gemini3 Pro. I respect his curse.

Draft Session: 40m.

![](https://i0.wp.com/teknotum.net/wp-content/uploads/2025/12/gemini_3_deep_think.jpg?fit=600%2C256&ssl=1)

## Where did my Critical Thinking go? 

### I'm still the Protagonist of my life.

After publishing this Kaggle Notebook and almost 24 hours of reflection I realized what happened here. 

I've just opened Google Gemini Studio Gallery page and asked one question. In only 22 seconds, Gemini3 Pro made the images and build the text with the respective Sources.

Sounds great, Gemini3 did the hard work and delivered it in a beautiful way.
Which was easy and fast. However, easy and fast isn't the best for All situations.

Our work starts when Gemini3's task is finished. That's the moment when Curiosity, Creativity and mostly Critical Thinking will divide the World. Those that are satisfied with just a Gemini3's presentation and those that will go further pursuing what remains from our Originality and Capacity to innovate, keeping the humanity progress. Also for many participants here, staying relevant in such a competitive world.

Anyway, be inspired by Ötzi. That man had skills, advanced tools and a sophisticated craftsmanship for his time, till dying from an arrow wound.
Therefore, never take anything for granted otherwise you could be put down by some magnificent AI.

At least, Ötzi's mummy was found to "tell" his history. What we'll find left of you to be told?