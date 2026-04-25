# santa-2025: cross-solution summary

This competition centered on geometric tree-packing and layout optimization, where participants aimed to minimize bounding dimensions while strictly enforcing overlap, symmetry, and scaling constraints. Winning approaches predominantly leveraged metaheuristic optimizers (e.g., genetic algorithms, Sparrow/Spurrow, simulated annealing, and memetic algorithms) combined with strategic seeding from crystal lattices and symmetry enforcement. Success heavily depended on GPU-accelerated relaxation, dynamic scaling across tree counts (N), and robust diversity management to escape local minima and converge to feasible zero-overlap solutions.

## Competition flows
- Raw tree shape data is processed into a geometric packing problem, optimized via a multi-island genetic algorithm with GPU-accelerated LBFGS relaxation and symmetry constraints, then finalized with a legalization step to produce the submission.
- The pipeline generates diverse rectangle seeds via lattice generation and vanilla Sparrow, recursively combines them into larger shapes, refines layouts using a custom Rust Sparrow optimizer with symmetry and orientation biases, and applies a fast C++ simulated annealing and compression step to minimize the bounding square size.
- Geometric constraints are addressed by generating initial tree layouts via concave hull approximation and Spyrrow/Sparrow, refining them with simulated annealing and memetic algorithms, and scaling across N values through layout transfer, culminating in final submitted configurations.
- Raw geometric constraints are initialized with hand-designed tile patterns, processed through `spyrrow` for initial placement, refined via a custom profile evaluator and simulated annealing, and exported as the final submission.

## Data processing
- Enforced 180° rotational symmetry for even tree counts
- Seeded solutions with tesselated crystal structures (scattering edge trees randomly)
- Applied mutation moves (Move, Jiggle, Twist, Translate)
- Minimized overlap via GPU relaxation
- Applied a final legalization step to achieve zero-overlap
- Lattice generation for rectangle bands
- Recursive combination of rectangles into larger shapes
- Up/down-sampling across different N values for seed generation
- Global rotations and slight scaling to 'kick' initial solutions
- C++ 'push' procedure to compress layouts by iteratively moving trees in predefined directions

## Models
- Sparrow algorithm
- simulated annealing (SA)

## Frameworks used
- CUDA
- PyTorch
- spyrrow
- sparrow
- alphashape

## Loss functions
- cost = α * ΣΣ overlap_ij^2 + β * Σ outside_i^2, where overlap uses exact separation distance via a 3D lookup table with trilinear interpolation

## Ensembling
- Managed a ring of solution islands that exchange lower-scoring champions to maintain diversity, followed by a final legalization step to convert low-overlap solutions into feasible zero-overlap submissions.

## Notable individual insights
- rank 1 (1st place: genetic algorithm and GPU relaxation): Enforcing 180° symmetry drastically reduces degrees of freedom and improves convergence for even tree counts.
- rank 1 (1st place: genetic algorithm and GPU relaxation): Precomputing separation distances in a 3D lookup table enables massive GPU relaxation throughput.
- rank 3 (Third place solution: a customized sparrow algorithm): Aligning the optimizer's objective directly with the competition metric (minimizing both width and height) is crucial for performance.
- rank 3 (Third place solution: a customized sparrow algorithm): Optimal solutions consistently exhibit a lattice-like structure with median tree angles clustering around 22-24° and 202-204°.
- rank 12 (Santa25 12th place): Replacing a dense tree lattice with a concave hull allows optimization libraries to treat it as a single object, significantly speeding up initial guess generation for large N.
- rank 8 (Santa25 8th Place): A custom profile evaluator for arbitrary shapes is more effective than `alphashape` for defining precise geometric boundaries in this packing problem.

## Solutions indexed
- #1 [[solutions/rank_01/solution|1st place: genetic algorithm and GPU relaxation]]
- #1 [[solutions/rank_01/solution|1st place solution preview]]
- #3 [[solutions/rank_03/solution|Third place solution: a customized sparrow algorithm]]
- #8 [[solutions/rank_08/solution|Santa25 8th Place]]
- #12 [[solutions/rank_12/solution|Santa25 12th place]]

## GitHub links
- [jcottaar/packing](https://github.com/jcottaar/packing) _(solution)_ — from [[solutions/rank_01/solution|1st place: genetic algorithm and GPU relaxation]]
- [PaulDL-RS/spyrrow](https://github.com/PaulDL-RS/spyrrow) _(library)_ — from [[solutions/rank_12/solution|Santa25 12th place]]
- [JeroenGar/sparrow](https://github.com/JeroenGar/sparrow) _(library)_ — from [[solutions/rank_12/solution|Santa25 12th place]]
