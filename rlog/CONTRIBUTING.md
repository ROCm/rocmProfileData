# Contributing to rlog

Contributions are welcome. This document covers how to build, test, and submit changes.

## Building and testing

Dependencies:
- CMake 3.14 or later
- A C++14-capable compiler
- `libsqlite3-dev`

```sh
mkdir build && cd build
cmake ..
make
ctest
```

Tests that require ROCTx or NVTX libraries skip automatically if those libraries are not installed.

## Submitting changes

- Open a GitHub issue before starting significant work, so we can discuss the approach.
- Fork the repository and submit a pull request against `main`.
- Keep commits focused; one logical change per PR where practical.
- Match the style of surrounding code. The project uses C++14.

## Code review

All pull requests require review from a code owner before merging. Reviewers will be assigned automatically via CODEOWNERS.

## Reporting issues

Use the GitHub issue tracker. Include steps to reproduce, observed behaviour, and expected behaviour.
