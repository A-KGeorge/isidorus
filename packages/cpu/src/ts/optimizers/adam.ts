import type { Tensor } from "@isidorus/core";
import { DType } from "@isidorus/core";
import type { Graph } from "../graph.js";
import type { Session } from "../session.js";
import {
  variableWithInit,
  globalVariablesInitializer,
  zerosInitializer,
  applyAdam,
  readVariable,
  assignVariable,
} from "../ops/variable_ops.js";
import { scalar } from "../ops/array_ops.js";
import { mul } from "../ops/math_ops.js";
import type { ParamSpec, FeedEntry } from "./sgd.js";

// ---------------------------------------------------------------------------
// Adam — Adaptive Moment Estimation (Kingma & Ba, 2015).
//
// Per-parameter update (bias-corrected):
//   m  = beta1 * m + (1 - beta1) * grad
//   v  = beta2 * v + (1 - beta2) * grad²
//   w  = w - lr * m̂ / (sqrt(v̂) + ε)
//
// TF's ResourceApplyAdam implements the full update in a single fused kernel.
// Beta powers (beta1^t, beta2^t) are tracked as scalar variables and decayed
// each step — the decay NoOp has control deps on all param update ops so it
// always runs after every parameter is updated.
//
// Usage:
//   const [dw, db] = g.addGradients([loss], [wRead, bRead]);
//   const opt = new Adam(g, [
//     { handle: wHandle, grad: dw, dtype: DType.FLOAT32, name: "w" },
//     { handle: bHandle, grad: db, dtype: DType.FLOAT32, name: "b" },
//   ], 0.001);
//   await opt.init(sess);
//   await opt.applyGradients(sess, [[x, feed]]);
// ---------------------------------------------------------------------------

export class Adam {
  private readonly g: Graph;
  private readonly lr: number;
  private readonly beta1: number;
  private readonly beta2: number;
  private readonly epsilon: number;

  private readonly allInitOps: string[];
  private initOpName: string | null = null;
  private initialised = false;

  // Single target that sequences all param updates + beta power decay.
  private readonly stepOp: string;

  constructor(
    g: Graph,
    params: ParamSpec[],
    lr = 0.001,
    options: { beta1?: number; beta2?: number; epsilon?: number } = {},
  ) {
    this.g = g;
    this.lr = lr;
    this.beta1 = options.beta1 ?? 0.9;
    this.beta2 = options.beta2 ?? 0.999;
    this.epsilon = options.epsilon ?? 1e-7;
    this.allInitOps = [];

    const lrT = scalar(g, this.lr);
    const b1T = scalar(g, this.beta1);
    const b2T = scalar(g, this.beta2);
    const epsT = scalar(g, this.epsilon);

    // ── Beta power variables ───────────────────────────────────────────────
    const b1Buf = Buffer.allocUnsafe(4);
    b1Buf.writeFloatLE(this.beta1, 0);
    const b2Buf = Buffer.allocUnsafe(4);
    b2Buf.writeFloatLE(this.beta2, 0);

    const { handle: b1Handle, initOp: b1Init } = variableWithInit(
      g,
      [],
      DType.FLOAT32,
      "adam/beta1_power",
      scalar(g, this.beta1),
    );
    const { handle: b2Handle, initOp: b2Init } = variableWithInit(
      g,
      [],
      DType.FLOAT32,
      "adam/beta2_power",
      scalar(g, this.beta2),
    );
    this.allInitOps.push(b1Init, b2Init);

    // ── Per-parameter moment variables + update ops ────────────────────────
    const paramUpdateOps: string[] = [];

    for (const { handle, grad, dtype, name } of params) {
      const shape = (grad.shape ?? []) as number[];

      const { handle: mHandle, initOp: mInit } = variableWithInit(
        g,
        shape,
        dtype,
        `${name}/adam_m`,
        zerosInitializer(g, shape, dtype),
      );
      const { handle: vHandle, initOp: vInit } = variableWithInit(
        g,
        shape,
        dtype,
        `${name}/adam_v`,
        zerosInitializer(g, shape, dtype),
      );
      this.allInitOps.push(mInit, vInit);

      // Read current beta powers — these are the same nodes read by every
      // param, which is correct: all params share the same step counter.
      const b1Power = readVariable(g, b1Handle, DType.FLOAT32);
      const b2Power = readVariable(g, b2Handle, DType.FLOAT32);

      const updateOp = applyAdam(
        g,
        handle,
        mHandle,
        vHandle,
        b1Power,
        b2Power,
        lrT,
        b1T,
        b2T,
        epsT,
        grad, // ← symbolic gradient wired directly, no placeholder
        dtype,
        `${name}/adam_update`,
      );
      paramUpdateOps.push(updateOp);
    }

    // ── Beta power decay — must run after ALL param updates ────────────────
    const b1Cur = readVariable(g, b1Handle, DType.FLOAT32);
    const b2Cur = readVariable(g, b2Handle, DType.FLOAT32);
    const b1Decay = assignVariable(
      g,
      b1Handle,
      mul(g, b1Cur, b1T),
      DType.FLOAT32,
      "adam/b1_decay",
    );
    const b2Decay = assignVariable(
      g,
      b2Handle,
      mul(g, b2Cur, b2T),
      DType.FLOAT32,
      "adam/b2_decay",
    );

    // stepOp sequences: all param updates, then both beta decays.
    this.stepOp = "adam/step";
    g.addOp("NoOp", [], {}, this.stepOp, [...paramUpdateOps, b1Decay, b2Decay]);
  }

  async init(sess: Session): Promise<void> {
    if (this.initialised) return;
    if (!this.initOpName)
      this.initOpName = globalVariablesInitializer(
        this.g,
        this.allInitOps,
        "adam_init",
      );
    await sess.run([], [], [this.initOpName]);
    this.initialised = true;
  }

  /**
   * targetOps — the op names that constitute one Adam step.
   * Exposed so trainStep can merge forward + backward into a single runAsync.
   */
  get targetOps(): string[] {
    return [this.stepOp];
  }

  /**
   * applyGradients — run one Adam step.
   * Runs param updates + beta power decay in a single TF_SessionRun.
   */
  async applyGradients(sess: Session, feeds: FeedEntry[]): Promise<void> {
    await sess.run(feeds, [], [this.stepOp]);
  }
}
