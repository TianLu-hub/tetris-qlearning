# Attribution and provenance

This is a new reconstruction of a Boston University CS 440 Tetris reinforcement-learning project. It was written from the assignment's interface requirements and checked against the locally supplied `tetris-1.0.3.jar`. It is not a recovery of the original 2025 source or trained weights.

The course game engine, neural-network primitives, matrix library and assignment were supplied by the course. Their original authors retain their rights. They are **not redistributed** in this repository; obtain the engine through an authorized course source and place it in `lib/`.

The setup script obtains [argparse4j 0.9.0](https://argparse4j.github.io/) from Maven Central and verifies its SHA-256. Argparse4j is a separate third-party dependency; its licensing information is included in its distribution. Downloaded JARs are excluded from Git.

The agent implementation, headless runner, tests, scripts and documentation in this repository are newly written reconstruction work. Scores in `results/` refer only to the explicitly recorded reconstruction runs, using the course engine's game score. They do not establish or reproduce historical resume claims.
