# Bronze medal border solution (finally in 92th place)

- **Author:** Koshiro Morioka
- **Date:** 2025-03-12T16:36:46.580Z
- **Topic ID:** 567893
- **URL:** https://www.kaggle.com/competitions/lux-ai-season-3/discussion/567893
---

Thank you for such an exciting competition. I would like to express my respect to the organizers and other participants. Just like the Chess competition, the two-week period where our bots compete against each other is truly thrilling.

My submission notebook is [here](https://www.kaggle.com/code/kosirowada/rulebaseagent)!

We participated in this competition by improving a public [notebook](https://www.kaggle.com/code/egrehbbt/relicbound-bot-for-lux-ai-s3-competition) created by @egrehbbt. This notebook was probably the most well-structured and strongest among the public codes available.

We also attempted imitation learning from top teams, but we struggled with managing the state properly, which resulted in units moving in only one direction. So, we had to abandon this approach.

We made three main modifications to the public notebook:

Modification 1:
When a reward or relic is newly discovered, all units are directed toward the unit that made the discovery.

Modification 2:
For game_number = 1, 2, 3, units that are not harvesting or searching for rewards or relics move randomly.
In the original public code, during the exploration phase of game_number = 1, 2, 3, units without the find_reward or find_relic task remained in the same place. To improve this, we moved those units toward unknown nodes, increasing the chances of discovering relics and rewards.

Modification 3: SAP Implementation
The original code did not implement sap. We introduced sap for units that are harvesting and still have remaining energy.
If an enemy enters within a Manhattan distance of 8, the unit will execute a sap attack.
If a moving unit uses sap, it might run out of energy before reaching its destination. To avoid this, we restricted sap usage to units that are harvesting.
Additionally, when executing a sap attack, the (x, y, z) parameters should be set so that y, z = (enemy's coordinates - current unit position). Thus, the enemy's raw coordinates should not be directly assigned to y, z.

With these modifications, we achieved a solution near the bronze medal threshold.
If we had implemented offensive strategies such as sending non-harvesting units to attack the enemy's reward zones, or making units move via energy-gaining points, we believe we could have developed an even stronger bot. We should have spent more time on this competition.

Thank you for reading this far. Let's watch how our bots perform over the next two weeks!