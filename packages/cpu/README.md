# @isidorus/cpu

High-performance TensorFlow CPU backend for Isidorus, enabling graph construction, training, and inference in Node.js environments.

## Why a Native Addon?

Integrating TensorFlow with Node.js isn't just about bindings—it's about solving fundamental scheduling conflicts:

- **libuv's Thread Pool** (Node.js event loop) and **TensorFlow's Scheduler** both compete for limited CPU resources
- Naive NAPI async worker wrapping leads to thread over-subscription, cache thrashing, and degraded latency
- This package uses intelligent request queueing with `uv_queue_work` to serialize inference and prevent scheduler conflicts
- The `reserveCores` option lets you control TensorFlow's resource usage relative to your event loop

The result: predictable latency and stable throughput, even under heavy inference load.

See the [main Isidorus README](../README.md#why-isidorus) for more context on the threading challenges this solves.

## Features

- **Native Addon**: Uses a C++ native addon to interface directly with `libtensorflow`.
- **Automatic Setup**: Automatically handles the download and installation of required TensorFlow libraries.
- **Inference Pool**: Efficiently manage concurrent execution strategies (worker-pool vs tf-parallel).
- **Ops Library**: Rich set of operations including Math, Array, NN, and Variable ops.
- **Smart Threading**: Avoids scheduler conflicts through coordinated resource management between libuv and TensorFlow threads.

## Installation

```bash
npm install @isidorus/cpu
```

_Note: The install script will automatically attempt to resolve and download `libtensorflow` if it is missing._

## Quick Start

```typescript
import { graph, session, ops, DType } from "@isidorus/cpu";

const g = graph();
const x = ops.placeholder(g, "x", [null, 784], DType.FLOAT32);

const sess = session(g, { reserveCores: 2 });
```

## Development

### Native Builds

To rebuild the native C++ addon:

```bash
npm run build:native
```

### Prebuilds

To generate prebuilt binaries for multiple Node.js versions:

```bash
npm run prebuildify
```
