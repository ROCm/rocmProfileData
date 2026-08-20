# Contributing to rocmProfileData

Contributions are welcome. This document covers how to build, test, and submit changes.

## Building

Dependencies are installed and the project is built via the provided script:

```sh
sudo bash install.sh
```

This installs system dependencies and builds all components. Components that require ROCm hardware (such as `rpd_tracer`) will build partially and skip GPU-dependent features if ROCm is not present.

## Submitting changes

- Open a GitHub issue before starting significant work, so we can discuss the approach.
- Fork the repository and submit a pull request against `main`.
- Keep commits focused; one logical change per PR where practical.

## Code review

All pull requests require review from a code owner before merging. Reviewers will be assigned automatically via CODEOWNERS.

## Reporting issues

Use the GitHub issue tracker. Include steps to reproduce, observed behaviour, and expected behaviour.
