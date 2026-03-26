# @isidorus/core

The foundational package for the Isidorus ecosystem, providing shared types and hardware-agnostic tensor abstractions.

Like its namesake, St. Isidore of Seville, this package serves as a central repository for the fundamental "knowledge" and structures (tensors and graphs) used throughout the library.

## Features

- **DType**: Comprehensive data type definitions and utilities.
- **Shape**: Utilities for tensor rank, element counting, and shape compatibility.
- **Tensor IR**: A standardized representation for tensors used across different execution backends.

## Installation

```bash
npm install @isidorus/core
```

## Usage

```typescript
import { DType, makeTensor, Shape } from "@isidorus/core";

const shape: Shape = [2, 2];
const tensor = makeTensor(shape, DType.FLOAT32, new Float32Array([1, 2, 3, 4]));
```

## Scripts

- `npm run build`: Compiles TypeScript to JavaScript using `tsc`.
- `npm test`: Runs the test suite using the internal test runner.
