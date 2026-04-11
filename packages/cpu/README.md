# @isidorus/cpu

High-performance TensorFlow CPU backend for Isidorus, enabling graph construction, training, and inference in Node.js environments.

## Features

- **Native Addon**: Uses a C++ native addon to interface directly with `libtensorflow`.
- **Automatic Setup**: Automatically handles the download and installation of required TensorFlow libraries.
- **Inference Pool**: Efficiently manage concurrent execution strategies (worker-pool vs tf-parallel).
- **Ops Library**: Rich set of operations including Math, Array, NN, and Variable ops.

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
