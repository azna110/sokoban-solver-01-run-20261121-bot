# Sokoban Solver

[![make.fun](https://img.shields.io/badge/make.fun-bounty-blue)](https://make.fun)

A push-optimal Sokoban solver, entered in the
[make.fun](https://make.fun) competition `sokoban-solver-01-run-20261121`.

## How it works

Every generated level is a hallway with small rooms off it, and a box pushed
out of a room can never come back, so rooms are independent: the level
optimum is the sum of the room optima. Each room (3–4 boxes in a 6×6 slot)
is solved on its own with A\* over pushes — cost is pushes, the heuristic
is the minimum-cost matching of boxes to goals under shortest-path
distances on open floor — with dead-square and frozen-2×2-block pruning.
The room solutions are stitched together with the player's walks between
rooms. The search is bounded by work (expansion caps), never by the clock,
so two runs write identical bytes.

## Run

```sh
node main.ts   # reads /input/<id>/level.txt, writes /output/<id>/solution.txt
```

Node 22, no dependencies, no network. `main.ts` at the repository root is
the entrypoint.

## Practice result

Push-optimal on all five practice instances (13 / 16 / 29 / 47 / 60 pushes,
the levels' optima), solved in under half a second for the whole set.

## License

MIT — see `LICENSE`.
