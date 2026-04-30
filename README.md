# Isidorus

Isidorus is a high-performance machine learning library for Node.js featuring TensorFlow-backed CPU execution, shared tensor IR, and graph abstractions.

The project is named in honor of **St. Isidore of Seville**, the patron saint of the internet and computer users, who was known for his efforts to compile and preserve the world's knowledge.

### For Production Edge ML

If you're deploying ML inference on edge devices with Node.js and need both:

- Predictable latency
- Stable throughput
- Non-blocking event loop behavior

Then Isidorus handles the hard scheduling problems you shouldn't have to solve yourself.

### Architecture & Concurrency

**Why the Event Loop Isn't Blocked During Inference:**

The computational heavy lifting is offloaded to background threads using Node.js's native C++ addon system and the `libuv` thread pool:

1. **Async Entry Point (`runAsync`):** When inference is requested, instead of executing synchronously, a standard JavaScript Promise is returned and execution is delegated to the native C++ module.
2. **Offloading to `libuv`:** The native C++ module (`SessionWrap::RunAsync`) prepares the TensorFlow inputs but uses `uv_queue_work` to schedule the actual execution (`TF_SessionRun`) on `libuv`'s background worker pool. The main Node.js event loop immediately resumes, allowing your app to continue processing network requests.
3. **Independent Threading:** `TF_SessionRun` executes in the background. TensorFlow uses its own highly optimized threading model (configured via `intraOpThreads`) to parallelize the math computations independent of Node.js.
4. **Thread-Safe Completion:** Once `TF_SessionRun` finishes, it packages the tensor outputs and pushes them to a thread-safe internal queue. It then calls a Thread-Safe Function (`napi_call_threadsafe_function`) to signal the main thread.
5. **Resolving the Promise:** The main event loop receives the signal, briefly drains the completion queue to construct the JavaScript arrays/tensors, and resolves the Javascript Promise you `await`ed earlier.

**What Happens at High Traffic in a Real-Time Server?**

Under high load (e.g., handling many concurrent WebSocket or HTTP requests that trigger model inference), raw TensorFlow bindings present two major threats:

1. **`libuv` Thread Pool Starvation:** By default, Node.js only has 4 worker threads in the `libuv` pool. If 4 concurrent inferences occur, they occupy all 4 background threads. The main JS event loop still ticks, but any other asynchronous Node.js operations that rely on `libuv` (like reading/writing files, certain cryptography, or DNS lookups) will be stalled in a queue until an inference finishes.
2. **CPU Thrashing:** If you try to fix starvation by increasing Node's `UV_THREADPOOL_SIZE=100` and process 100 concurrent jobs, you create a new problem. If each inference uses 4 CPU cores (`intraOpThreads`), you are requesting 400 highly active threads on a machine with limited physical cores. The OS scheduler will thrash (forcefully pausing and resuming threads), resulting in astronomical latency spikes for all requests.

**The Solution: `InferencePool`**
To prevent starvation and thrashing, Isidorus provides an `InferencePool` class that protects the system:

- **Concurrency Clamping (`maxConcurrent`):** It restricts the number of active `runAsync()` calls allowed in-flight at any given moment, strictly limiting them based on your physical CPU cores and the `UV_THREADPOOL_SIZE`.
- **JS-level Queueing:** If a new request arrives while the concurrency limit is maxed out, the `InferencePool` intercepts it and holds it in a standard JavaScript Array on the main thread.
- **Graceful Degradation:** Excess requests politely wait their turn. As load spikes, overall latency increases linearly (due to queueing), but the server's CPU remains at peak efficiency, context switching overhead is avoided, and Node.js stays highly responsive to standard I/O.

**How does `InferencePool` know the optimal `maxConcurrent` value?**

Regardless of configuration, the pool always calculates two absolute hardware limitations first:

1.  **Physical Cores (`usable`):** It queries the OS for the number of physical CPU cores (or uses a strictly defined number if `reserveCores` is set).
2.  **`libuv` Thread Pool Limit (`poolCap`):** It checks `process.env.UV_THREADPOOL_SIZE` natively and unconditionally caps `maxConcurrent` so it never exceeds the worker threads actually available to Node.js.

Then, the `InferencePool` settles on the final `maxConcurrent` capability via one of three routes:

- **A. Expert Mode (Explicit Configuration):** If you explicitly pass `maxConcurrent` or `intraOpThreads`, it derives the missing value using the formula `Math.floor(usable / provided_value)` to balance threads safely without oversubscribing.
- **B. Pre-defined Profiles:**
  - `profile="latency"` allocates all `usable` cores to a single inference (`intra=usable`, `maxConcurrent=1`) to optimize for the lowest single-request time.
  - `profile="throughput"` statically locks threads per job to 4 (`intra=4`) and runs exactly `Math.floor(usable / 4)` jobs concurrently.
- **C. The Autotuner (Default):** If no profile or strict numbers are set, the pool runs an internal benchmark pipeline at startup:
  1.  **Generate Candidates:** It creates a list of potential thread configurations (e.g., 2 `intra` cores, 4 `intra` cores, etc.).
  2.  **Warm-up:** It runs inferences using dummy tensors that exactly match your model's input shapes to pre-heat TensorFlow and oneDNN caches.
  3.  **Stress Test:** It blasts the worker pool with continuous dummy inference calls for multiple cycles exactly at the concurrency cap, measuring the sustained requests per second (RPS) capability.
  4.  **Pick Winner:** It selects the configuration that yielded the highest overall RPS. If two configurations tie, it smartly prefers the one with higher `intra` threads so that individual request latency remains lower while still providing maximum throughput.

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
