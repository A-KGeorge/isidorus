# Isidorus

Isidorus is a high-performance machine learning library for Node.js featuring TensorFlow-backed CPU execution, shared tensor IR, and graph abstractions.

The project is named in honor of **St. Isidore of Seville**, the patron saint of the internet and computer users, who was known for his efforts to compile and preserve the world's knowledge.

## Why Isidorus?

### The Threading Over-Subscription Problem

Integrating TensorFlow with Node.js requires solving a fundamental scheduling conflict. Both libuv (Node's event loop) and TensorFlow's thread scheduler are **greedy for CPU resources**, leading to thread over-subscription on edge devices with limited cores.

### Naive Approaches Don't Work

Simply wrapping TensorFlow operations in NAPI async workers (as other solutions attempt) creates several performance problems:

1. **Thread Contention**: Both schedulers compete for L1/L2 cache, creating a "noisy neighbors" situation
2. **Latency Degradation**: While throughput may appear unchanged, latency increases significantly due to cache coherency overhead
3. **Event Loop Blocking**: TensorFlow's greedy scheduling can starve the main event loop during inference bursts

This is why libraries like SciPy in Python also suffer when multithreaded at the application level—you hit the same architectural limitation.

### Isidorus's Solution

Isidorus uses **auto-tuning and intelligent request queueing** with `uv_queue_work` to:

- **Limit TensorFlow's resource footprint** relative to libuv's static thread pool
- **Increase the libuv thread pool size** to accommodate both base worker threads and TensorFlow threads
- **Serialize requests intelligently** to prevent scheduler conflicts and cache thrashing
- **Maintain event loop responsiveness** even during heavy inference workloads

The key insight: understand the underlying engine (libuv + TensorFlow's scheduling internals) rather than layering abstractions that fight each other. This approach is similar to how Triton solves this for GPU workloads, but optimized for CPU-bound edge inference.

### For Production Edge ML

If you're deploying ML inference on edge devices with Node.js and need both:

- Predictable latency
- Stable throughput
- Non-blocking event loop behavior

Then Isidorus handles the hard scheduling problems you shouldn't have to solve yourself.

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
