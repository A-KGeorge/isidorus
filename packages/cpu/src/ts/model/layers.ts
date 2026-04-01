import type { Tensor } from "@isidorus/core";
import { DType } from "@isidorus/core";
import type { Graph } from "../graph.js";
import type { Layer, LayerParam, ActivationFn } from "./layer.js";
import {
  variableWithInit,
  readVariable,
  zerosInitializer,
  glorotUniformInitializer,
  truncatedNormalInitializer,
} from "../ops/variable_ops.js";
import { constant } from "../ops/array_ops.js";
import { matmul, biasAdd } from "../ops/math_ops.js";
import {
  relu,
  leakyRelu,
  relu6,
  sigmoid,
  conv2d as conv2dOp,
  tanh,
  softmax,
  logSoftmax,
  elu,
  selu,
  swish,
  gelu,
} from "../ops/nn_ops.js";

// ---------------------------------------------------------------------------
// Activation helper
// ---------------------------------------------------------------------------
function activate(g: Graph, x: Tensor, fn: ActivationFn, name: string): Tensor {
  switch (fn) {
    case "relu":
      return relu(g, x, `${name}/relu`);
    case "leaky_relu":
      return leakyRelu(g, x, 0.2, `${name}/leaky_relu`);
    case "relu6":
      return relu6(g, x, `${name}/relu6`);
    case "sigmoid":
      return sigmoid(g, x, `${name}/sigmoid`);
    case "tanh":
      return tanh(g, x, `${name}/tanh`);
    case "softmax":
      return softmax(g, x, `${name}/softmax`);
    case "log_softmax":
      return logSoftmax(g, x, `${name}/log_softmax`);
    case "elu":
      return elu(g, x, `${name}/elu`);
    case "selu":
      return selu(g, x, `${name}/selu`);
    case "swish":
      return swish(g, x, `${name}/swish`);
    case "gelu":
      return gelu(g, x, `${name}/gelu`);
    case "linear":
      return x;
    default:
      throw new Error(`Unknown activation: ${fn}`);
  }
}

// ---------------------------------------------------------------------------
// Dense — fully-connected layer: output = activation(input @ W + b)
// ---------------------------------------------------------------------------
export class Dense implements Layer {
  readonly name: string;
  output!: Tensor;
  readonly layerParams: LayerParam[] = [];

  private readonly units: number;
  private readonly activation: ActivationFn;
  private readonly useBias: boolean;

  constructor(
    units: number,
    options: {
      activation?: ActivationFn;
      useBias?: boolean;
      name?: string;
    } = {},
  ) {
    this.units = units;
    this.activation = options.activation ?? "linear";
    this.useBias = options.useBias ?? true;
    this.name = options.name ?? `dense_${units}`;
  }

  build(
    g: Graph,
    input: Tensor,
    inputShape: (number | null)[],
  ): (number | null)[] {
    const inFeatures = inputShape[inputShape.length - 1] as number;
    if (!inFeatures || inFeatures < 1)
      throw new Error(
        `Dense "${
          this.name
        }": last input dim must be known, got ${JSON.stringify(inputShape)}`,
      );

    // ── Weight W: [inFeatures, units] ────────────────────────────────────
    const wInitVal = glorotUniformInitializer(
      g,
      [inFeatures, this.units],
      DType.FLOAT32,
      `${this.name}/w_glorot`,
    );
    const { handle: wHandle, initOp: wInitOp } = variableWithInit(
      g,
      [inFeatures, this.units],
      DType.FLOAT32,
      `${this.name}/w`,
      wInitVal,
    );
    const wRead = readVariable(
      g,
      wHandle,
      DType.FLOAT32,
      `${this.name}/w_read`,
    );

    let out = matmul(g, input, wRead, {}, `${this.name}/matmul`);
    this.layerParams.push({
      handle: wHandle,
      read: wRead,
      dtype: DType.FLOAT32,
      name: `${this.name}/w`,
      initOp: wInitOp,
    });

    // ── Bias b: [units] ───────────────────────────────────────────────────
    if (this.useBias) {
      const bInitVal = zerosInitializer(g, [this.units], DType.FLOAT32);
      const { handle: bHandle, initOp: bInitOp } = variableWithInit(
        g,
        [this.units],
        DType.FLOAT32,
        `${this.name}/b`,
        bInitVal,
      );
      const bRead = readVariable(
        g,
        bHandle,
        DType.FLOAT32,
        `${this.name}/b_read`,
      );
      out = biasAdd(g, out, bRead, `${this.name}/bias_add`);
      this.layerParams.push({
        handle: bHandle,
        read: bRead,
        dtype: DType.FLOAT32,
        name: `${this.name}/b`,
        initOp: bInitOp,
      });
    }

    this.output = activate(g, out, this.activation, this.name);
    return [...inputShape.slice(0, -1), this.units];
  }
}

// ---------------------------------------------------------------------------
// Flatten — reshapes [batch, d1, d2, ...] → [batch, d1*d2*...]
// ---------------------------------------------------------------------------
export class Flatten implements Layer {
  readonly name: string;
  output!: Tensor;
  readonly layerParams: LayerParam[] = [];

  constructor(options: { name?: string } = {}) {
    this.name = options.name ?? "flatten";
  }

  build(
    g: Graph,
    input: Tensor,
    inputShape: (number | null)[],
  ): (number | null)[] {
    const spatialDims = inputShape.slice(1);
    const hasUnknown = spatialDims.some((d) => d === null);
    const flatSize = hasUnknown
      ? null
      : spatialDims.reduce((a, b) => a! * b!, 1);
    const flatDim = flatSize ?? -1;

    const shapeBuf = Buffer.allocUnsafe(8);
    shapeBuf.writeInt32LE(-1, 0);
    shapeBuf.writeInt32LE(flatDim, 4);
    const shapeConst = constant(
      g,
      shapeBuf,
      [2],
      DType.INT32,
      `${this.name}/shape`,
    );
    const [out] = g.addOp(
      "Reshape",
      [input, shapeConst],
      {},
      `${this.name}/reshape`,
    );
    this.output = out;
    return [null, flatSize];
  }
}

// ---------------------------------------------------------------------------
// Conv2D — 2D convolution (NHWC): output = activation(conv2d(input, W) + b)
// ---------------------------------------------------------------------------
export class Conv2D implements Layer {
  readonly name: string;
  output!: Tensor;
  readonly layerParams: LayerParam[] = [];

  private readonly filters: number;
  private readonly kernelSize: [number, number];
  private readonly strides: [number, number, number, number];
  private readonly padding: "SAME" | "VALID";
  private readonly activation: ActivationFn;
  private readonly useBias: boolean;

  constructor(
    filters: number,
    options: {
      kernelSize?: number | [number, number];
      strides?: number | [number, number];
      padding?: "SAME" | "VALID";
      activation?: ActivationFn;
      useBias?: boolean;
      name?: string;
    } = {},
  ) {
    this.filters = filters;
    this.padding = options.padding ?? "SAME";
    this.activation = options.activation ?? "linear";
    this.useBias = options.useBias ?? true;
    this.name = options.name ?? `conv2d_${filters}f`;

    const ks = options.kernelSize ?? 3;
    this.kernelSize = Array.isArray(ks) ? (ks as [number, number]) : [ks, ks];

    const st = options.strides ?? 1;
    const [sH, sW] = Array.isArray(st) ? (st as [number, number]) : [st, st];
    this.strides = [1, sH, sW, 1];
  }

  build(
    g: Graph,
    input: Tensor,
    inputShape: (number | null)[],
  ): (number | null)[] {
    if (inputShape.length !== 4)
      throw new Error(
        `Conv2D "${this.name}": expects 4D input [batch,H,W,C], got rank ${inputShape.length}`,
      );

    const inChannels = inputShape[3] as number;
    if (!inChannels || inChannels < 1)
      throw new Error(
        `Conv2D "${this.name}": in_channels must be known, got ${JSON.stringify(
          inputShape,
        )}`,
      );

    const [kH, kW] = this.kernelSize;
    const wShape = [kH, kW, inChannels, this.filters];

    // He normal init for conv: stddev = sqrt(2 / (kH * kW * inChannels))
    const stddev = Math.sqrt(2 / (kH * kW * inChannels));
    const wInitVal = truncatedNormalInitializer(
      g,
      wShape,
      DType.FLOAT32,
      { stddev },
      `${this.name}/w_init`,
    );
    const { handle: wHandle, initOp: wInitOp } = variableWithInit(
      g,
      wShape,
      DType.FLOAT32,
      `${this.name}/w`,
      wInitVal,
    );
    const wRead = readVariable(
      g,
      wHandle,
      DType.FLOAT32,
      `${this.name}/w_read`,
    );

    let out = conv2dOp(
      g,
      input,
      wRead,
      this.strides,
      this.padding,
      `${this.name}/conv`,
    );
    this.layerParams.push({
      handle: wHandle,
      read: wRead,
      dtype: DType.FLOAT32,
      name: `${this.name}/w`,
      initOp: wInitOp,
    });

    if (this.useBias) {
      const bInitVal = zerosInitializer(g, [this.filters], DType.FLOAT32);
      const { handle: bHandle, initOp: bInitOp } = variableWithInit(
        g,
        [this.filters],
        DType.FLOAT32,
        `${this.name}/b`,
        bInitVal,
      );
      const bRead = readVariable(
        g,
        bHandle,
        DType.FLOAT32,
        `${this.name}/b_read`,
      );
      out = biasAdd(g, out, bRead, `${this.name}/bias_add`);
      this.layerParams.push({
        handle: bHandle,
        read: bRead,
        dtype: DType.FLOAT32,
        name: `${this.name}/b`,
        initOp: bInitOp,
      });
    }

    this.output = activate(g, out, this.activation, this.name);

    // Output spatial dimensions
    const [, H, W] = inputShape;
    const [, sH, sW] = this.strides;
    const outH =
      H === null
        ? null
        : this.padding === "SAME"
        ? Math.ceil(H / sH)
        : Math.ceil((H - kH + 1) / sH);
    const outW =
      W === null
        ? null
        : this.padding === "SAME"
        ? Math.ceil(W / sW)
        : Math.ceil((W - kW + 1) / sW);

    return [null, outH, outW, this.filters];
  }
}
