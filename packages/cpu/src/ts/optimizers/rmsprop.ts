import type { Tensor } from "@isidorus/core";
import { DType } from "@isidorus/core";
import type { Graph } from "../graph.js";
import type { Session } from "../session.js";
import {
  variableWithInit,
  globalVariablesInitializer,
  zerosInitializer,
  applyRMSProp,
} from "../ops/variable_ops.js";
import { scalar } from "../ops/array_ops.js";
import type { ParamSpec, FeedEntry } from "./sgd.js";

// ---------------------------------------------------------------------------
// RMSProp — Root Mean Squared Propagation (Hinton, 2012).
//
// Per-parameter update:
//   ms  = rho * ms + (1 - rho) * grad²
//   mom = momentum * mom + lr * grad / sqrt(ms + ε)
//   w   = w - mom
//
// Usage:
//   const [dw] = g.addGradients([loss], [wRead]);
//   const opt = new RMSProp(g, [
//     { handle: wHandle, grad: dw, dtype: DType.FLOAT32, name: "w" },
//   ], 0.001);
//   await opt.init(sess);
//   await opt.applyGradients(sess, [[x, feed]]);
// ---------------------------------------------------------------------------

export class RMSProp {
  private readonly g: Graph;
  private readonly lr: number;
  private readonly rho: number;
  private readonly momentum: number;
  private readonly epsilon: number;

  private readonly allInitOps: string[];
  private initOpName: string | null = null;
  private initialised = false;

  private readonly updateOps: string[];

  constructor(
    g: Graph,
    params: ParamSpec[],
    lr = 0.001,
    options: { rho?: number; momentum?: number; epsilon?: number } = {},
  ) {
    this.g = g;
    this.lr = lr;
    this.rho = options.rho ?? 0.9;
    this.momentum = options.momentum ?? 0.0;
    this.epsilon = options.epsilon ?? 1e-7;
    this.allInitOps = [];
    this.updateOps = [];

    const lrT = scalar(g, this.lr);
    const rhoT = scalar(g, this.rho);
    const momT = scalar(g, this.momentum);
    const epsT = scalar(g, this.epsilon);

    for (const { handle, grad, dtype, name } of params) {
      const shape = (grad.shape ?? []) as number[];

      const { handle: msHandle, initOp: msInit } = variableWithInit(
        g,
        shape,
        dtype,
        `${name}/rms_ms`,
        zerosInitializer(g, shape, dtype),
      );
      const { handle: momHandle, initOp: momInit } = variableWithInit(
        g,
        shape,
        dtype,
        `${name}/rms_mom`,
        zerosInitializer(g, shape, dtype),
      );
      this.allInitOps.push(msInit, momInit);

      const updateOp = applyRMSProp(
        g,
        handle,
        msHandle,
        momHandle,
        lrT,
        rhoT,
        momT,
        epsT,
        grad, // ← symbolic gradient, no placeholder
        dtype,
        `${name}/rms_update`,
      );
      this.updateOps.push(updateOp);
    }
  }

  async init(sess: Session): Promise<void> {
    if (this.initialised) return;
    if (this.allInitOps.length > 0) {
      if (!this.initOpName)
        this.initOpName = globalVariablesInitializer(
          this.g,
          this.allInitOps,
          "rmsprop_init",
        );
      await sess.run([], [], [this.initOpName]);
    }
    this.initialised = true;
  }

  async applyGradients(sess: Session, feeds: FeedEntry[]): Promise<void> {
    await sess.run(feeds, [], this.updateOps);
  }
}
