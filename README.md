# Isidorus

Isidorus is a high-performance machine learning library for Node.js featuring TensorFlow-backed CPU execution, shared tensor IR, and graph abstractions.

The project is named in honor of **St. Isidore of Seville**, the patron saint of the internet and computer users, who was known for his efforts to compile and preserve the world's knowledge.

## Project Structure

This is a monorepo managed with npm workspaces and [Changesets](https://github.com/changesets/changesets).

- `packages/core`: Shared types, Tensor Intermediate Representation (IR), and graph abstractions.
- `packages/cpu`: TensorFlow CPU graph construction, training, and inference for Node.js.

## Getting Started

### Prerequisites

- Node.js (v18, v20, v22, or v24 recommended for native compatibility)
- TypeScript ^5.4.0

### Installation

```bash
npm install
```

### Build

Build all packages in the workspace:

```bash
npm run build
```

### Testing

Run tests across all packages:

```bash
npm test
```

## Versioning and Publishing

This project uses `@changesets/cli` for versioning.

- To create a new version: `npm run changeset`
- To bump versions: `npm run version`
- To publish to npm: `npm run publish-packages`

## License

[Apache-2.0](https://www.google.com/search?q=LICENSE)
