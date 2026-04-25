# 65th~ place solution just on Rules and will to experiment

- **Author:** Pizzaboi
- **Date:** 2025-03-10T23:59:49.160Z
- **Topic ID:** 567567
- **URL:** https://www.kaggle.com/competitions/lux-ai-season-3/discussion/567567

**GitHub links found:**
- https://github.com/ArtemVeshkin/luxai-s3

---

It was a fun competition! This was my first experience with simulation challenges, and I'm glad I teamed up with @veshkinartem.

Huge thanks to the organizers and special appreciation to @egrehbbt, whose excellent starter kernel provided a solid foundation for our solution.

Our solution: [GitHub](https://github.com/ArtemVeshkin/luxai-s3)

Key areas I'd love to see explored in other write-ups:

Reinforcement Learning approaches and practical tips for successful implementation.

Techniques for global optimization of sapling distribution across ships, avoiding greedy approaches.

Insights into how top teams estimated the DROP_OFF_FACTOR.

Our most advanced bot can be found at merged_3 folder.

Our improvements over public bot of @egrehbbt:
1. Nebula gas estimation energy drop
2. Sap logic with shared  damage distribution
3. Optimized harvesting logic
4. Barrier building to defend our farming spots using non-employed ships
5. Empiric prediction of enemy ship movement
6. Loosing logic: if we lose, then we can sacrifice exploration and focus on harvesting at the end of round 

Things which we tried and which did not really work:
1. Global sap optimization along all ships together
2. Prediction of obstacles movement in a pathfinding
3. Enemy ramming logic
4. Energy interpolation with U-Net. Worked well as isolated time, but gave no boost while really playing using it's predictions
5. We also tried implementing this article: https://arxiv.org/pdf/2301.01609
It really looked good for this environment, but in reality was struggling to learn
