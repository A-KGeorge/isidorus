import type { Tensor } from "@isidorus/core";
import { DType } from "@isidorus/core";

/**
 * Options for Session construction.
 *
 * Thread counts control how TensorFlow uses CPU cores internally.
 * The right values depend on how many concurrent runAsync() calls you expect:
 *
 *   Single inference (Worker thread):
 *     intraOpThreads: 0, interOpThreads: 0  — let TF use all cores
 *
 *   Concurrent runAsync() on main thread (default):
 *     intraOpThreads: 2, interOpThreads: 2  — share cores across requests
 *
 *   Formula for N concurrent requests on C cores:
 *     intraOpThreads = Math.max(1, Math.floor(C / N))
 *     interOpThreads = Math.max(1, Math.floor(C / N))
 */
export interface SessionOptions {
  /**
   * Per-op parallelism — threads used within a single op (e.g. matmul tiles).
   * Default: 2. Set to 0 to let TF choose automatically.
   */
  intraOpThreads?: number;

  /**
   * Graph-level parallelism — threads used to run independent ops concurrently.
   * Default: 2. Set to 0 to let TF choose automatically.
   */
  interOpThreads?: number;

  /**
   * The NUMA node layout to pin this session's threads to.
   * Defaults to undefined (disabled) - handled internally by OS scheduling.
   */
  numaNode?: number;
}

/** Raw tensor value passed as a feed. */
export interface FeedValue {
  dtype: DType;
  shape: number[];
  data: Buffer | Uint8Array;
}

/** Result from a fetched tensor output. */
export interface TensorValue {
  dtype: DType;
  shape: number[];
  data: Buffer;
}

/** Convert a Tensor to the native feed format. */
function toNativeFeed(t: Tensor, v: FeedValue) {
  return {
    opName: t.opName,
    index: t.index,
    tensor: {
      dtype: Number(v.dtype),
      shape: v.shape,
      data: Buffer.isBuffer(v.data)
        ? v.data
        : Buffer.from(v.data.buffer, v.data.byteOffset, v.data.byteLength),
    },
  };
}

/** Convert a Tensor to the native fetch format. */
function toNativeFetch(t: Tensor) {
  return { opName: t.opName, index: t.index };
}

/** Parse native output back to TensorValue. */
function fromNativeOutput(raw: any): TensorValue {
  return {
    dtype: raw.dtype as DType,
    shape: raw.shape as number[],
    data: raw.data as Buffer,
  };
}

/**
 * Session — executes a Graph.
 *
 * A Session holds a TF_Session backed by the graph passed at construction.
 * Once created the graph's structure should not change (new ops may be added
 * for initialisation ops, but existing ops should not be removed).
 *
 * @example
 * const sess = new Session(graph);
 * await sess.run([], [], ["init"]);          // run init op (side-effect)
 * const [output] = await sess.runAsync(     // fetch result
 *   [[x, inputFeed]],
 *   [y],
 * );
 */
export class Session {
  /** @internal */
  readonly _native: any;

  constructor(native: any) {
    this._native = native;
  }

  /**
   * Synchronous inference — blocks the event loop during TF_SessionRun.
   * Use on Worker threads. On the main thread prefer runAsync().
   *
   * @param feeds    Array of [Tensor, FeedValue] pairs
   * @param fetches  Tensors to compute and return
   * @param targets  Op names to run for side-effects only (e.g. "init_all")
   */
  async run(
    feeds: [Tensor, FeedValue][],
    fetches: Tensor[],
    targets: string[] = [],
  ): Promise<TensorValue[]> {
    const nativeFeeds = feeds.map(([t, v]) => toNativeFeed(t, v));
    const nativeFetches = fetches.map(toNativeFetch);
    const raw = (await this._native.run(
      nativeFeeds,
      nativeFetches,
      targets,
    )) as any[];
    return raw.map(fromNativeOutput);
  }

  /**
   * Non-blocking inference — TF_SessionRun on the libuv thread pool.
   * The event loop stays free for I/O and timers during compute.
   *
   * @param feeds    Array of [Tensor, FeedValue] pairs
   * @param fetches  Tensors to compute and return
   * @param targets  Op names to run for side-effects only
   */
  async runAsync(
    feeds: [Tensor, FeedValue][],
    fetches: Tensor[],
    targets: string[] = [],
  ): Promise<TensorValue[]> {
    const nativeFeeds = feeds.map(([t, v]) => toNativeFeed(t, v));
    const nativeFetches = fetches.map(toNativeFetch);
    const raw = (await this._native.runAsync(
      nativeFeeds,
      nativeFetches,
      targets,
    )) as any[];
    return raw.map(fromNativeOutput);
  }

  /** Close the session and release all C++ resources. */
  destroy(): void {
    this._native.destroy();
  }
}
