import type { Tensor } from "@isidorus/core";
import { DType } from "@isidorus/core";
import type { Graph } from "../graph.js";
import type {
  Layer,
  LayerParam,
  ActivationFn,
  WeightMap,
  DenseConfig,
  Conv2DConfig,
  FlattenConfig,
  DepthwiseConv2DConfig,
  SeparableConv2DConfig,
  MaxPooling2DConfig,
  GlobalAveragePooling2DConfig,
  ZeroPadding2DConfig,
  BatchNormalizationConfig,
} from "./layer.js";
import {
  variableWithInit,
  readVariable,
  assignVariable,
  zerosInitializer,
  onesInitializer,
  glorotUniformInitializer,
  truncatedNormalInitializer,
} from "../ops/variable_ops.js";
import { constant, pad } from "../ops/array_ops.js";
import { matmul, mul, add, biasAdd } from "../ops/math_ops.js";
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
  depthwiseConv2d,
  maxPool,
  globalAvgPool,
  batchNorm,
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
  private inFeatures: number = 0; // Set during build()

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
    this.inFeatures = inFeatures; // Store for paramCount()
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

  buildFrozen(
    g: Graph,
    input: Tensor,
    inputShape: (number | null)[],
    weights: WeightMap,
  ): { tensor: Tensor; shape: (number | null)[] } {
    const wEntry = weights.get(`${this.name}/w`);
    if (!wEntry)
      throw new Error(`Dense.buildFrozen: missing weight "${this.name}/w"`);

    const wConst = constant(
      g,
      wEntry.data,
      wEntry.shape,
      wEntry.dtype,
      `${this.name}/w`,
    );
    let out = matmul(g, input, wConst, {}, `${this.name}/matmul`);

    if (this.useBias) {
      const bEntry = weights.get(`${this.name}/b`);
      if (!bEntry)
        throw new Error(`Dense.buildFrozen: missing weight "${this.name}/b"`);
      const bConst = constant(
        g,
        bEntry.data,
        bEntry.shape,
        bEntry.dtype,
        `${this.name}/b`,
      );
      out = biasAdd(g, out, bConst, `${this.name}/bias_add`);
    }

    const tensor = activate(g, out, this.activation, this.name);
    return { tensor, shape: [...inputShape.slice(0, -1), this.units] };
  }

  getConfig(): DenseConfig {
    return {
      type: "Dense",
      name: this.name,
      units: this.units,
      activation: this.activation,
      useBias: this.useBias,
    };
  }

  paramCount(): number {
    // Weight matrix: [inFeatures, units] + bias: [units]
    let count = this.inFeatures * this.units;
    if (this.useBias) count += this.units;
    return count;
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

  buildFrozen(
    g: Graph,
    input: Tensor,
    inputShape: (number | null)[],
    _weights: WeightMap,
  ): { tensor: Tensor; shape: (number | null)[] } {
    // Flatten has no weights — identical to build() except returns tensor directly.
    const spatialDims = inputShape.slice(1);
    const flatSize = spatialDims.some((d) => d === null)
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
    const [tensor] = g.addOp(
      "Reshape",
      [input, shapeConst],
      {},
      `${this.name}/reshape`,
    );
    return { tensor, shape: [null, flatSize] };
  }

  getConfig(): FlattenConfig {
    return { type: "Flatten", name: this.name };
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
  private inChannels: number = 0; // Set during build()

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
    this.inChannels = inChannels; // Store for paramCount()
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

  buildFrozen(
    g: Graph,
    input: Tensor,
    inputShape: (number | null)[],
    weights: WeightMap,
  ): { tensor: Tensor; shape: (number | null)[] } {
    const wEntry = weights.get(`${this.name}/w`);
    if (!wEntry)
      throw new Error(`Conv2D.buildFrozen: missing weight "${this.name}/w"`);

    const wConst = constant(
      g,
      wEntry.data,
      wEntry.shape,
      wEntry.dtype,
      `${this.name}/w`,
    );
    let out = conv2dOp(
      g,
      input,
      wConst,
      this.strides,
      this.padding,
      `${this.name}/conv`,
    );

    if (this.useBias) {
      const bEntry = weights.get(`${this.name}/b`);
      if (!bEntry)
        throw new Error(`Conv2D.buildFrozen: missing weight "${this.name}/b"`);
      const bConst = constant(
        g,
        bEntry.data,
        bEntry.shape,
        bEntry.dtype,
        `${this.name}/b`,
      );
      out = biasAdd(g, out, bConst, `${this.name}/bias_add`);
    }

    const tensor = activate(g, out, this.activation, this.name);

    const [, H, W] = inputShape;
    const [kH, kW] = this.kernelSize;
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

    return { tensor, shape: [null, outH, outW, this.filters] };
  }

  getConfig(): Conv2DConfig {
    return {
      type: "Conv2D",
      name: this.name,
      filters: this.filters,
      kernelSize: this.kernelSize,
      strides: [this.strides[1], this.strides[2]], // extract sH, sW from [1,sH,sW,1]
      padding: this.padding,
      activation: this.activation,
      useBias: this.useBias,
    };
  }

  paramCount(): number {
    // Kernel: [kH, kW, inChannels, filters] + bias: [filters]
    const [kH, kW] = this.kernelSize;
    let count = kH * kW * this.inChannels * this.filters;
    if (this.useBias) count += this.filters;
    return count;
  }
}

// ---------------------------------------------------------------------------
// DepthwiseConv2D — applies a depthwise filter per input channel.
//   output channels = in_channels * depth_multiplier
// ---------------------------------------------------------------------------
export class DepthwiseConv2D implements Layer {
  readonly name: string;
  output!: Tensor;
  readonly layerParams: LayerParam[] = [];

  private readonly kernelSize: [number, number];
  private readonly strides: [number, number, number, number];
  private readonly padding: "SAME" | "VALID";
  private readonly depthMultiplier: number;
  private readonly activation: ActivationFn;
  private readonly useBias: boolean;

  constructor(
    options: {
      kernelSize?: number | [number, number];
      strides?: number | [number, number];
      padding?: "SAME" | "VALID";
      depthMultiplier?: number;
      activation?: ActivationFn;
      useBias?: boolean;
      name?: string;
    } = {},
  ) {
    this.depthMultiplier = options.depthMultiplier ?? 1;
    this.padding = options.padding ?? "SAME";
    this.activation = options.activation ?? "linear";
    this.useBias = options.useBias ?? true;
    this.name = options.name ?? "depthwise_conv2d";

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
    const inC = inputShape[3] as number;
    const [kH, kW] = this.kernelSize;
    const wShape = [kH, kW, inC, this.depthMultiplier];
    const stddev = Math.sqrt(2 / (kH * kW * inC));

    const wInit = truncatedNormalInitializer(
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
      wInit,
    );
    const wRead = readVariable(
      g,
      wHandle,
      DType.FLOAT32,
      `${this.name}/w_read`,
    );
    this.layerParams.push({
      handle: wHandle,
      read: wRead,
      dtype: DType.FLOAT32,
      name: `${this.name}/w`,
      initOp: wInitOp,
    });

    let out = depthwiseConv2d(
      g,
      input,
      wRead,
      this.strides,
      this.padding,
      `${this.name}/dw`,
    );

    if (this.useBias) {
      const outC = inC * this.depthMultiplier;
      const bInit = zerosInitializer(g, [outC], DType.FLOAT32);
      const { handle: bHandle, initOp: bInitOp } = variableWithInit(
        g,
        [outC],
        DType.FLOAT32,
        `${this.name}/b`,
        bInit,
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
    return [null, outH, outW, inC * this.depthMultiplier];
  }

  buildFrozen(
    g: Graph,
    input: Tensor,
    inputShape: (number | null)[],
    weights: WeightMap,
  ): { tensor: Tensor; shape: (number | null)[] } {
    const wEntry = weights.get(`${this.name}/w`)!;
    const inC = inputShape[3] as number;
    const [kH, kW] = this.kernelSize;
    const wConst = constant(
      g,
      wEntry.data,
      wEntry.shape,
      wEntry.dtype,
      `${this.name}/w`,
    );
    let out = depthwiseConv2d(
      g,
      input,
      wConst,
      this.strides,
      this.padding,
      `${this.name}/dw`,
    );
    if (this.useBias) {
      const bEntry = weights.get(`${this.name}/b`)!;
      const bConst = constant(
        g,
        bEntry.data,
        bEntry.shape,
        bEntry.dtype,
        `${this.name}/b`,
      );
      out = biasAdd(g, out, bConst, `${this.name}/bias_add`);
    }
    const tensor = activate(g, out, this.activation, this.name);
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
    return { tensor, shape: [null, outH, outW, inC * this.depthMultiplier] };
  }

  getConfig(): DepthwiseConv2DConfig {
    return {
      type: "DepthwiseConv2D",
      name: this.name,
      depthMultiplier: this.depthMultiplier,
      kernelSize: this.kernelSize,
      strides: [this.strides[1], this.strides[2]],
      padding: this.padding,
      activation: this.activation,
      useBias: this.useBias,
    };
  }
}

// ---------------------------------------------------------------------------
// SeparableConv2D — depthwise + pointwise convolution.
//   Params: depthwise kernel [kH,kW,inC,depthMul] + pointwise kernel [1,1,inC*depthMul,filters]
// ---------------------------------------------------------------------------
export class SeparableConv2D implements Layer {
  readonly name: string;
  output!: Tensor;
  readonly layerParams: LayerParam[] = [];

  private readonly filters: number;
  private readonly kernelSize: [number, number];
  private readonly strides: [number, number, number, number];
  private readonly padding: "SAME" | "VALID";
  private readonly depthMultiplier: number;
  private readonly activation: ActivationFn;
  private readonly useBias: boolean;

  constructor(
    filters: number,
    options: {
      kernelSize?: number | [number, number];
      strides?: number | [number, number];
      padding?: "SAME" | "VALID";
      depthMultiplier?: number;
      activation?: ActivationFn;
      useBias?: boolean;
      name?: string;
    } = {},
  ) {
    this.filters = filters;
    this.depthMultiplier = options.depthMultiplier ?? 1;
    this.padding = options.padding ?? "SAME";
    this.activation = options.activation ?? "linear";
    this.useBias = options.useBias ?? true;
    this.name = options.name ?? `sep_conv2d_${filters}f`;

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
    const inC = inputShape[3] as number;
    const [kH, kW] = this.kernelSize;
    const dwShape = [kH, kW, inC, this.depthMultiplier];
    const pwShape = [1, 1, inC * this.depthMultiplier, this.filters];

    // Depthwise kernel
    const dwInit = truncatedNormalInitializer(
      g,
      dwShape,
      DType.FLOAT32,
      { stddev: 0.1 },
      `${this.name}/dw_init`,
    );
    const { handle: dwH, initOp: dwInitOp } = variableWithInit(
      g,
      dwShape,
      DType.FLOAT32,
      `${this.name}/dw`,
      dwInit,
    );
    const dwRead = readVariable(g, dwH, DType.FLOAT32, `${this.name}/dw_read`);
    this.layerParams.push({
      handle: dwH,
      read: dwRead,
      dtype: DType.FLOAT32,
      name: `${this.name}/dw`,
      initOp: dwInitOp,
    });

    // Pointwise kernel
    const pwInit = glorotUniformInitializer(
      g,
      pwShape,
      DType.FLOAT32,
      `${this.name}/pw_init`,
    );
    const { handle: pwH, initOp: pwInitOp } = variableWithInit(
      g,
      pwShape,
      DType.FLOAT32,
      `${this.name}/pw`,
      pwInit,
    );
    const pwRead = readVariable(g, pwH, DType.FLOAT32, `${this.name}/pw_read`);
    this.layerParams.push({
      handle: pwH,
      read: pwRead,
      dtype: DType.FLOAT32,
      name: `${this.name}/pw`,
      initOp: pwInitOp,
    });

    let out = depthwiseConv2d(
      g,
      input,
      dwRead,
      this.strides,
      this.padding,
      `${this.name}/dw_conv`,
    );
    out = conv2dOp(
      g,
      out,
      pwRead,
      [1, 1, 1, 1],
      "SAME",
      `${this.name}/pw_conv`,
    );

    if (this.useBias) {
      const bInit = zerosInitializer(g, [this.filters], DType.FLOAT32);
      const { handle: bH, initOp: bInitOp } = variableWithInit(
        g,
        [this.filters],
        DType.FLOAT32,
        `${this.name}/b`,
        bInit,
      );
      const bRead = readVariable(g, bH, DType.FLOAT32, `${this.name}/b_read`);
      out = biasAdd(g, out, bRead, `${this.name}/bias`);
      this.layerParams.push({
        handle: bH,
        read: bRead,
        dtype: DType.FLOAT32,
        name: `${this.name}/b`,
        initOp: bInitOp,
      });
    }

    this.output = activate(g, out, this.activation, this.name);

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

  buildFrozen(
    g: Graph,
    input: Tensor,
    inputShape: (number | null)[],
    weights: WeightMap,
  ): { tensor: Tensor; shape: (number | null)[] } {
    const dw = weights.get(`${this.name}/dw`)!;
    const pw = weights.get(`${this.name}/pw`)!;
    const [kH, kW] = this.kernelSize;
    let out = depthwiseConv2d(
      g,
      input,
      constant(g, dw.data, dw.shape, dw.dtype, `${this.name}/dw`),
      this.strides,
      this.padding,
      `${this.name}/dw_conv`,
    );
    out = conv2dOp(
      g,
      out,
      constant(g, pw.data, pw.shape, pw.dtype, `${this.name}/pw`),
      [1, 1, 1, 1],
      "SAME",
      `${this.name}/pw_conv`,
    );
    if (this.useBias) {
      const b = weights.get(`${this.name}/b`)!;
      out = biasAdd(
        g,
        out,
        constant(g, b.data, b.shape, b.dtype, `${this.name}/b`),
        `${this.name}/bias`,
      );
    }
    const tensor = activate(g, out, this.activation, this.name);
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
    return { tensor, shape: [null, outH, outW, this.filters] };
  }

  getConfig(): SeparableConv2DConfig {
    return {
      type: "SeparableConv2D",
      name: this.name,
      filters: this.filters,
      kernelSize: this.kernelSize,
      strides: [this.strides[1], this.strides[2]],
      padding: this.padding,
      depthMultiplier: this.depthMultiplier,
      activation: this.activation,
      useBias: this.useBias,
    };
  }
}

// ---------------------------------------------------------------------------
// MaxPooling2D
// ---------------------------------------------------------------------------
export class MaxPooling2D implements Layer {
  readonly name: string;
  output!: Tensor;
  readonly layerParams: LayerParam[] = [];

  private readonly poolSize: [number, number];
  private readonly strides: [number, number];
  private readonly padding: "SAME" | "VALID";

  constructor(
    options: {
      poolSize?: number | [number, number];
      strides?: number | [number, number];
      padding?: "SAME" | "VALID";
      name?: string;
    } = {},
  ) {
    const ps = options.poolSize ?? 2;
    this.poolSize = Array.isArray(ps) ? (ps as [number, number]) : [ps, ps];
    const st = options.strides ?? this.poolSize;
    this.strides = Array.isArray(st) ? (st as [number, number]) : [st, st];
    this.padding = options.padding ?? "VALID";
    this.name = options.name ?? "max_pool2d";
  }

  build(
    g: Graph,
    input: Tensor,
    inputShape: (number | null)[],
  ): (number | null)[] {
    const [pH, pW] = this.poolSize;
    const [sH, sW] = this.strides;
    this.output = maxPool(
      g,
      input,
      [1, pH, pW, 1],
      [1, sH, sW, 1],
      this.padding,
      this.name,
    );
    const [, H, W, C] = inputShape;
    const outH =
      H === null
        ? null
        : this.padding === "SAME"
        ? Math.ceil(H / sH)
        : Math.floor((H - pH) / sH) + 1;
    const outW =
      W === null
        ? null
        : this.padding === "SAME"
        ? Math.ceil(W / sW)
        : Math.floor((W - pW) / sW) + 1;
    return [null, outH, outW, C];
  }

  buildFrozen(
    g: Graph,
    input: Tensor,
    inputShape: (number | null)[],
    _w: WeightMap,
  ): { tensor: Tensor; shape: (number | null)[] } {
    const [pH, pW] = this.poolSize;
    const [sH, sW] = this.strides;
    const tensor = maxPool(
      g,
      input,
      [1, pH, pW, 1],
      [1, sH, sW, 1],
      this.padding,
      this.name,
    );
    const [, H, W, C] = inputShape;
    const outH =
      H === null
        ? null
        : this.padding === "SAME"
        ? Math.ceil(H / sH)
        : Math.floor((H - pH) / sH) + 1;
    const outW =
      W === null
        ? null
        : this.padding === "SAME"
        ? Math.ceil(W / sW)
        : Math.floor((W - pW) / sW) + 1;
    return { tensor, shape: [null, outH, outW, C] };
  }

  getConfig(): MaxPooling2DConfig {
    return {
      type: "MaxPooling2D",
      name: this.name,
      poolSize: this.poolSize,
      strides: this.strides,
      padding: this.padding,
    };
  }
}

// ---------------------------------------------------------------------------
// GlobalAveragePooling2D — reduces [N,H,W,C] → [N,C] via spatial mean.
// ---------------------------------------------------------------------------
export class GlobalAveragePooling2D implements Layer {
  readonly name: string;
  output!: Tensor;
  readonly layerParams: LayerParam[] = [];

  constructor(options: { name?: string } = {}) {
    this.name = options.name ?? "global_avg_pool";
  }

  build(
    g: Graph,
    input: Tensor,
    inputShape: (number | null)[],
  ): (number | null)[] {
    this.output = globalAvgPool(g, input, this.name);
    return [null, inputShape[3] ?? null];
  }

  buildFrozen(
    g: Graph,
    input: Tensor,
    inputShape: (number | null)[],
    _w: WeightMap,
  ): { tensor: Tensor; shape: (number | null)[] } {
    return {
      tensor: globalAvgPool(g, input, this.name),
      shape: [null, inputShape[3] ?? null],
    };
  }

  getConfig(): GlobalAveragePooling2DConfig {
    return { type: "GlobalAveragePooling2D", name: this.name };
  }
}

// ---------------------------------------------------------------------------
// ZeroPadding2D — pads the spatial dimensions of a 4D tensor with zeros.
// ---------------------------------------------------------------------------
export class ZeroPadding2D implements Layer {
  readonly name: string;
  output!: Tensor;
  readonly layerParams: LayerParam[] = [];

  private readonly padding: [[number, number], [number, number]];

  constructor(
    options: {
      padding?: number | [[number, number], [number, number]];
      name?: string;
    } = {},
  ) {
    const p = options.padding ?? 1;
    this.padding =
      typeof p === "number"
        ? [
            [p, p],
            [p, p],
          ]
        : (p as [[number, number], [number, number]]);
    this.name = options.name ?? "zero_pad2d";
  }

  build(
    g: Graph,
    input: Tensor,
    inputShape: (number | null)[],
  ): (number | null)[] {
    const [[top, bot], [left, right]] = this.padding;
    this.output = pad(
      g,
      input,
      [
        [0, 0],
        [top, bot],
        [left, right],
        [0, 0],
      ],
      this.name,
    );
    const [, H, W, C] = inputShape;
    return [
      null,
      H === null ? null : H + top + bot,
      W === null ? null : W + left + right,
      C,
    ];
  }

  buildFrozen(
    g: Graph,
    input: Tensor,
    inputShape: (number | null)[],
    _w: WeightMap,
  ): { tensor: Tensor; shape: (number | null)[] } {
    const [[top, bot], [left, right]] = this.padding;
    const tensor = pad(
      g,
      input,
      [
        [0, 0],
        [top, bot],
        [left, right],
        [0, 0],
      ],
      this.name,
    );
    const [, H, W, C] = inputShape;
    return {
      tensor,
      shape: [
        null,
        H === null ? null : H + top + bot,
        W === null ? null : W + left + right,
        C,
      ],
    };
  }

  getConfig(): ZeroPadding2DConfig {
    return { type: "ZeroPadding2D", name: this.name, padding: this.padding };
  }
}

// ---------------------------------------------------------------------------
// BatchNormalization — training mode with EMA updates.
//
// Forward pass (training): uses batch mean/variance via FusedBatchNormV3
//   with is_training=true. After each step, moving_mean and moving_var are
//   updated via exponential moving average as side-effect ops.
//
// Forward pass (inference / exportFrozen): uses saved moving statistics,
//   built into the frozen graph as constants via buildFrozen().
//
// Trainable params:  gamma (scale), beta (offset)
// Non-trainable:     moving_mean, moving_var (updated by EMA, not gradients)
//
// Note on predict(): Sequential.predict() runs the training-path graph,
//   so it uses batch statistics (is_training=true). For correct inference
//   results use exportFrozen() → InferencePool.create().
// ---------------------------------------------------------------------------
export class BatchNormalization implements Layer {
  readonly name: string;
  output!: Tensor;
  readonly layerParams: LayerParam[] = []; // gamma, beta — gradient-trained
  private readonly _movingParams: LayerParam[] = []; // moving_mean, moving_var
  private readonly _updateOpNames: string[] = []; // EMA assign ops

  private readonly epsilon: number;
  private readonly momentum: number;

  constructor(
    options: { epsilon?: number; momentum?: number; name?: string } = {},
  ) {
    this.epsilon = options.epsilon ?? 1e-3;
    this.momentum = options.momentum ?? 0.99;
    this.name = options.name ?? "batch_norm";
  }

  build(
    g: Graph,
    input: Tensor,
    inputShape: (number | null)[],
  ): (number | null)[] {
    const channels = inputShape[inputShape.length - 1] as number;

    // ── Trainable params ──────────────────────────────────────────────────────

    // gamma (scale) — init to 1
    const { handle: gammaH, initOp: gammaInitOp } = variableWithInit(
      g,
      [channels],
      DType.FLOAT32,
      `${this.name}/gamma`,
      onesInitializer(g, [channels], DType.FLOAT32),
    );
    const gammaRead = readVariable(
      g,
      gammaH,
      DType.FLOAT32,
      `${this.name}/gamma_read`,
    );
    this.layerParams.push({
      handle: gammaH,
      read: gammaRead,
      dtype: DType.FLOAT32,
      name: `${this.name}/gamma`,
      initOp: gammaInitOp,
    });

    // beta (offset) — init to 0
    const { handle: betaH, initOp: betaInitOp } = variableWithInit(
      g,
      [channels],
      DType.FLOAT32,
      `${this.name}/beta`,
      zerosInitializer(g, [channels], DType.FLOAT32),
    );
    const betaRead = readVariable(
      g,
      betaH,
      DType.FLOAT32,
      `${this.name}/beta_read`,
    );
    this.layerParams.push({
      handle: betaH,
      read: betaRead,
      dtype: DType.FLOAT32,
      name: `${this.name}/beta`,
      initOp: betaInitOp,
    });

    // ── Non-trainable moving statistics ───────────────────────────────────────

    // moving_mean — init to 0
    const { handle: mmH, initOp: mmInitOp } = variableWithInit(
      g,
      [channels],
      DType.FLOAT32,
      `${this.name}/moving_mean`,
      zerosInitializer(g, [channels], DType.FLOAT32),
    );
    const mmRead = readVariable(g, mmH, DType.FLOAT32, `${this.name}/mm_read`);
    this._movingParams.push({
      handle: mmH,
      read: mmRead,
      dtype: DType.FLOAT32,
      name: `${this.name}/moving_mean`,
      initOp: mmInitOp,
    });

    // moving_var — init to 1
    const { handle: mvH, initOp: mvInitOp } = variableWithInit(
      g,
      [channels],
      DType.FLOAT32,
      `${this.name}/moving_var`,
      onesInitializer(g, [channels], DType.FLOAT32),
    );
    const mvRead = readVariable(g, mvH, DType.FLOAT32, `${this.name}/mv_read`);
    this._movingParams.push({
      handle: mvH,
      read: mvRead,
      dtype: DType.FLOAT32,
      name: `${this.name}/moving_var`,
      initOp: mvInitOp,
    });

    // ── Training-mode BN (FusedBatchNormV3, is_training=true) ─────────────────
    // Uses batch statistics for normalisation. Outputs:
    //   [0] y          — normalised tensor (routed to next layer)
    //   [1] batch_mean — used for EMA update below
    //   [2] batch_var  — used for EMA update below
    //   [3..5] reserve_space — for gradient computation, unused here
    const [y, batchMean, batchVar] = g.addOp(
      "FusedBatchNormV3",
      [input, gammaRead, betaRead, mmRead, mvRead],
      {
        epsilon: { kind: "float", value: this.epsilon },
        is_training: { kind: "bool", value: true },
        data_format: { kind: "string", value: "NHWC" },
      },
      `${this.name}/bn`,
    );

    // ── EMA update ops: run every training step as side effects ───────────────
    // new_mm = momentum * moving_mean + (1 - momentum) * batch_mean
    // new_mv = momentum * moving_var  + (1 - momentum) * batch_var
    const momBuf = Buffer.allocUnsafe(4);
    momBuf.writeFloatLE(this.momentum, 0);
    const oomBuf = Buffer.allocUnsafe(4);
    oomBuf.writeFloatLE(1 - this.momentum, 0);
    const mom = constant(g, momBuf, [], DType.FLOAT32, `${this.name}/ema_mom`);
    const oom = constant(g, oomBuf, [], DType.FLOAT32, `${this.name}/ema_oom`);

    // Second read ops — separate graph nodes to avoid aliasing with the BN inputs
    const mm2 = readVariable(g, mmH, DType.FLOAT32, `${this.name}/mm_ema_read`);
    const mv2 = readVariable(g, mvH, DType.FLOAT32, `${this.name}/mv_ema_read`);

    const newMM = add(
      g,
      mul(g, mm2, mom),
      mul(g, batchMean, oom),
      `${this.name}/new_mm`,
    );
    const newMV = add(
      g,
      mul(g, mv2, mom),
      mul(g, batchVar, oom),
      `${this.name}/new_mv`,
    );

    this._updateOpNames.push(
      assignVariable(g, mmH, newMM, DType.FLOAT32, `${this.name}/update_mm`),
      assignVariable(g, mvH, newMV, DType.FLOAT32, `${this.name}/update_mv`),
    );

    this.output = activate(g, y, this.activation ?? "linear", this.name);
    return inputShape;
  }

  buildFrozen(
    g: Graph,
    input: Tensor,
    inputShape: (number | null)[],
    weights: WeightMap,
  ): { tensor: Tensor; shape: (number | null)[] } {
    // Inference path: uses saved moving statistics as constants.
    const get = (n: string) => {
      const e = weights.get(n);
      if (!e) throw new Error(`BatchNorm.buildFrozen: missing weight "${n}"`);
      return e;
    };
    const gamma = get(`${this.name}/gamma`);
    const beta = get(`${this.name}/beta`);
    const mean_ = get(`${this.name}/moving_mean`);
    const var_ = get(`${this.name}/moving_var`);
    const tensor = batchNorm(
      g,
      input,
      constant(g, gamma.data, gamma.shape, gamma.dtype, `${this.name}/gamma`),
      constant(g, beta.data, beta.shape, beta.dtype, `${this.name}/beta`),
      constant(g, mean_.data, mean_.shape, mean_.dtype, `${this.name}/mean`),
      constant(g, var_.data, var_.shape, var_.dtype, `${this.name}/var`),
      { epsilon: this.epsilon },
      this.name,
    );
    return { tensor, shape: inputShape };
  }

  /** Op names to run each training step (EMA updates for moving stats). */
  updateOps(): string[] {
    return this._updateOpNames;
  }

  /** Non-trainable parameters — included in saveWeights but not in model.params. */
  nonTrainableParams(): LayerParam[] {
    return this._movingParams;
  }

  getConfig(): BatchNormalizationConfig {
    return {
      type: "BatchNormalization",
      name: this.name,
      epsilon: this.epsilon,
      momentum: this.momentum,
    };
  }

  // BatchNormalization has no useBias/activation in the constructor for now;
  // activation is always linear (BN is usually followed by an explicit activation layer).
  private readonly activation: "linear" = "linear";
}
