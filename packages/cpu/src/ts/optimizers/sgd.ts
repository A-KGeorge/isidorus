import type { Tensor } from "@isidorus/core";
import { DType } from "@isidorus/core";
import type { Graph } from "../graph.js";
import type { Session } from "../session.js";
import {
  variableWithInit,
  globalVariablesInitializer,
  zerosInitializer,
  applyGradientDescent,
} from "../ops/variable_ops.js";
import { scalar } from "../ops/array_ops.js";
import { mul, sub, add } from "../ops/math_ops.js";
import { readVariable, assignVariable } from "../ops/variable_ops.js";

// ---------------------------------------------------------------------------
// SGD — Stochastic Gradient Descent with optional momentum.
//
// Design:
//   Gradient tensors are wired into the update ops at construction time.
//   applyGradients simply runs the update op targets with the data feeds.
//   No placeholder indirection — the graph already references the grad tensors
//   returned by g.addGradients(), so running any update op also runs the
//   gradient computation automatically.
//
// Without momentum:
//   w = w - lr * grad
//
// With momentum (Polyak):
//   v = momentum * v - lr * grad
//   w = w + v
//
// Usage:
//   // 1. Build graph
//   const [dw, db] = g.addGradients([loss], [wRead, bRead]);
//
//   // 2. Create optimizer — pass grad tensors here
//   const opt = new SGD(g, [
//     { handle: wHandle, grad: dw, dtype: DType.FLOAT32, name: "w" },
//     { handle: bHandle, grad: db, dtype: DType.FLOAT32, name: "b" },
//   ], 0.01);
//
//   // 3. Init (only needed when momentum > 0)
//   await opt.init(sess);
//
//   // 4. Training loop
//   await opt.applyGradients(sess, [[x, xFeed], [y, yFeed]]);
// ---------------------------------------------------------------------------

export interface ParamSpec {
  /** VarHandleOp tensor from ops.variable() or ops.variableWithInit(). */
  handle: Tensor;
  /** Gradient tensor from g.addGradients() — dLoss/dParam. */
  grad: Tensor;
  /** Dtype of this parameter. */
  dtype: DType;
  /** Name used to generate op names for update ops. */
  name: string;
}

export type FeedEntry = [
  Tensor,
  { dtype: DType; shape: number[]; data: Buffer },
];

export class SGD {
  private readonly g: Graph;
  private readonly lr: number;
  private readonly momentum: number;

  private readonly updateOps: string[];
  private readonly velInitOps: string[];
  private initOpName: string | null = null;
  private initialised = false;

  constructor(
    g: Graph,
    params: ParamSpec[],
    lr: number,
    options: { momentum?: number } = {},
  ) {
    this.g = g;
    this.lr = lr;
    this.momentum = options.momentum ?? 0;
    this.updateOps = [];
    this.velInitOps = [];

    const lrT = scalar(g, lr);

    for (const { handle, grad, dtype, name } of params) {
      if (this.momentum === 0) {
        // ResourceApplyGradientDescent: w -= lr * grad
        // grad is the symbolic tensor from addGradients — wired directly.
        const op = applyGradientDescent(
          g,
          handle,
          lrT,
          grad,
          dtype,
          `${name}/sgd_update`,
        );
        this.updateOps.push(op);
      } else {
        const shape = (grad.shape ?? []) as number[];
        const momT = scalar(g, this.momentum);

        const { handle: velHandle, initOp: velInit } = variableWithInit(
          g,
          shape,
          dtype,
          `${name}/velocity`,
          zerosInitializer(g, shape, dtype),
        );
        this.velInitOps.push(velInit);

        // v = momentum * v - lr * grad
        const vCur = readVariable(g, velHandle, dtype);
        const vNew = sub(g, mul(g, momT, vCur), mul(g, lrT, grad));
        const vAssign = assignVariable(
          g,
          velHandle,
          vNew,
          dtype,
          `${name}/vel_assign`,
        );

        // w = w + v_new   (control dep: velocity assigned first)
        const wCur = readVariable(g, handle, dtype);
        const wNew = add(g, wCur, vNew);
        const wAssign = assignVariable(
          g,
          handle,
          wNew,
          dtype,
          `${name}/sgd_update`,
        );

        // Sequence v-assign → w-assign via a controlling NoOp.
        g.addOp("NoOp", [], {}, `${name}/sgd_seq`, [vAssign, wAssign]);
        this.updateOps.push(`${name}/sgd_seq`);
      }
    }
  }

  /**
   * init — initialise momentum velocity variables.
   * Must be called once before the first step when momentum > 0.
   * No-op when momentum === 0.
   */
  async init(sess: Session): Promise<void> {
    if (this.initialised) return;
    if (this.velInitOps.length > 0) {
      if (!this.initOpName)
        this.initOpName = globalVariablesInitializer(
          this.g,
          this.velInitOps,
          "sgd_vel_init",
        );
      await sess.run([], [], [this.initOpName]);
    }
    this.initialised = true;
  }

  /**
   * applyGradients — run one parameter update step.
   *
   * Runs all update ops as session targets. Because the update ops reference
   * the gradient tensors from addGradients(), TF automatically runs the
   * forward pass + backward pass to compute the gradients as part of the
   * same TF_SessionRun call.
   *
   * @param sess   Session to execute on
   * @param feeds  Data feeds — input tensors and labels for this batch
   */
  async applyGradients(sess: Session, feeds: FeedEntry[]): Promise<void> {
    await sess.run(feeds, [], this.updateOps);
  }
}
