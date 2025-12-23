# Contributing to Train-LLE

Thank you for your interest in contributing! We welcome pull requests, bug reports, and feature requests.

## Code of Conduct

Please note that this project is released with a [Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

## Development Workflow

1.  **Fork** the repository.
2.  **Clone** your fork locally.
3.  Install dependencies: `pnpm install`.
4.  Create a **branch** for your feature: `git checkout -b feature/my-new-feature`.
5.  Make your changes and add tests.
6.  Run tests: `pnpm test`.
7.  Commit your changes using conventional commits (e.g., `feat: add new layer`).
8.  Push to your fork and submit a **Pull Request**.

## Project Structure

- `packages/train-lle`: The main library source code.
- `archive/`: Archived legacy code (Python, JS-only packages).
- `examples/`: Usage examples.

## Style Guide

- Use TypeScript.
- Follow existing naming conventions.
- Format code using the project's Prettier/ESLint settings.
