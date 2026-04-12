/**
 * model/easy-layers.ts — Compound (multi-sub-layer) building blocks.
 *
 * Each class implements the Layer interface so it composes seamlessly with
 * Sequential and Model. Internally they hold arrays of primitive layers and
 * delegate graph construction to them, aggregating params and EMA ops.
 *
 * Available:
 *   ConvBnRelu       — Conv2D → BatchNorm → ReLU (common CNN pattern)
 *   ConvBnRelu6      — Conv2D → BatchNorm → ReLU6 (MobileNet pattern)
 *   InvertedResidual — MobileNetV2 bottleneck (expand → depthwise → project)
 *   ResidualBlock    — ResNet bottleneck (1×1 → 3×3 → 1×1 + skip)
 */

import type { Tensor } from "@isidorus/core";
import { DType } from "@isidorus/core";
import type { Graph } from "../graph.js";
import type { Layer, LayerParam, WeightMap, LayerConfig } from "./layer.js";
import { Conv2D, DepthwiseConv2D, BatchNormalization } from "./layers.js";
import { relu, relu6 } from "../ops/nn_ops.js";
import { add } from "../ops/math_ops.js";

// ── Base for compound layers ──────────────────────────────────────────────────

abstract class CompoundLayer implements Layer {
  abstract readonly name: string;
  output!: Tensor;
  readonly layerParams: LayerParam[] = [];

  protected sublayers: Layer[] = [];

  /** Build all sub-layers sequentially, propagating shape and collecting params. */
  protected buildSublayers(
    g: Graph,
    input: Tensor,
    inputShape: (number | null)[],
  ): (number | null)[] {
    let current = input;
    let shape = inputShape;
    for (const layer of this.sublayers) {
      shape = layer.build(g, current, shape);
      current = layer.output;
      this.layerParams.push(...layer.layerParams);
    }
    return shape;
  }

  /** Aggregate EMA update ops from BN sub-layers. */
  updateOps(): string[] {
    return this.sublayers.flatMap((l) => l.updateOps?.() ?? []);
  }

  /** Aggregate non-trainable params (BN moving stats) from sub-layers. */
  nonTrainableParams(): LayerParam[] {
    return this.sublayers.flatMap((l) => l.nonTrainableParams?.() ?? []);
  }

  abstract build(
    g: Graph,
    input: Tensor,
    inputShape: (number | null)[],
  ): (number | null)[];
  abstract buildFrozen(
    g: Graph,
    input: Tensor,
    inputShape: (number | null)[],
    weights: WeightMap,
  ): { tensor: Tensor; shape: (number | null)[] };

  abstract getConfig(): LayerConfig;
}

// ── ConvBnRelu ────────────────────────────────────────────────────────────────

/**
 * Conv2D → BatchNormalization → ReLU
 *
 * The most common CNN pattern. Fuses three layers into one composable unit.
 *
 * @example
 * new ConvBnRelu(64, 3, { stride: 2, padding: "SAME" })
 */
export class ConvBnRelu extends CompoundLayer {
  readonly name: string;
  private readonly filters: number;
  private readonly kernelSize: number;
  private readonly stride: number;
  private readonly padding: "SAME" | "VALID";

  constructor(
    filters: number,
    kernelSize: number,
    opts: {
      stride?: number;
      padding?: "SAME" | "VALID";
      name?: string;
    } = {},
  ) {
    super();
    this.filters = filters;
    this.kernelSize = kernelSize;
    this.stride = opts.stride ?? 1;
    this.padding = opts.padding ?? "SAME";
    this.name = opts.name ?? `conv_bn_relu_${filters}x${kernelSize}`;
  }

  build(
    g: Graph,
    input: Tensor,
    inputShape: (number | null)[],
  ): (number | null)[] {
    this.sublayers = [
      new Conv2D(this.filters, {
        kernelSize: this.kernelSize,
        strides: this.stride,
        padding: this.padding,
        useBias: false, // BN subsumes the bias shift
        activation: "linear",
        name: `${this.name}/conv`,
      }),
      new BatchNormalization({ name: `${this.name}/bn` }),
    ];

    let shape = this.buildSublayers(g, input, inputShape);
    // Apply ReLU to BN output — a simple graph op, not a Layer.
    this.output = relu(g, this.sublayers[1].output, `${this.name}/relu`);
    return shape;
  }

  buildFrozen(
    g: Graph,
    input: Tensor,
    inputShape: (number | null)[],
    weights: WeightMap,
  ): { tensor: Tensor; shape: (number | null)[] } {
    let current = input;
    let shape = inputShape;
    for (const l of this.sublayers) {
      const res = l.buildFrozen(g, current, shape, weights);
      current = res.tensor;
      shape = res.shape;
    }
    return { tensor: relu(g, current, `${this.name}/relu`), shape };
  }

  getConfig(): LayerConfig {
    // Use Conv2D config as the closest primitive match; compound type is not
    // in the LayerConfig union yet — add a ConvBnReluConfig union member if
    // round-trip serialisation via Sequential.loadModel() is required.
    return {
      type: "ConvBnRelu",
      name: this.name,
      filters: this.filters,
      kernelSize: this.kernelSize,
      stride: this.stride,
      padding: this.padding,
    } as any;
  }
}

// ── ConvBnRelu6 ────────────────────────────────────────────────────────────────

/**
 * Conv2D → BatchNormalization → ReLU6
 *
 * Preferred for MobileNet-style architectures where activation saturation
 * must be bounded for fixed-point quantization.
 */
export class ConvBnRelu6 extends CompoundLayer {
  readonly name: string;
  private readonly filters: number;
  private readonly kernelSize: number;
  private readonly stride: number;
  private readonly padding: "SAME" | "VALID";

  constructor(
    filters: number,
    kernelSize: number,
    opts: {
      stride?: number;
      padding?: "SAME" | "VALID";
      name?: string;
    } = {},
  ) {
    super();
    this.filters = filters;
    this.kernelSize = kernelSize;
    this.stride = opts.stride ?? 1;
    this.padding = opts.padding ?? "SAME";
    this.name = opts.name ?? `conv_bn_relu6_${filters}x${kernelSize}`;
  }

  build(
    g: Graph,
    input: Tensor,
    inputShape: (number | null)[],
  ): (number | null)[] {
    this.sublayers = [
      new Conv2D(this.filters, {
        kernelSize: this.kernelSize,
        strides: this.stride,
        padding: this.padding,
        useBias: false,
        activation: "linear",
        name: `${this.name}/conv`,
      }),
      new BatchNormalization({ name: `${this.name}/bn` }),
    ];

    let shape = this.buildSublayers(g, input, inputShape);
    this.output = relu6(g, this.sublayers[1].output, `${this.name}/relu6`);
    return shape;
  }

  buildFrozen(
    g: Graph,
    input: Tensor,
    inputShape: (number | null)[],
    weights: WeightMap,
  ): { tensor: Tensor; shape: (number | null)[] } {
    let current = input;
    let shape = inputShape;
    for (const l of this.sublayers) {
      const res = l.buildFrozen(g, current, shape, weights);
      current = res.tensor;
      shape = res.shape;
    }
    return { tensor: relu6(g, current, `${this.name}/relu6`), shape };
  }

  getConfig(): LayerConfig {
    return {
      type: "ConvBnRelu6",
      name: this.name,
      filters: this.filters,
      kernelSize: this.kernelSize,
      stride: this.stride,
      padding: this.padding,
    } as any;
  }
}

// ── InvertedResidual ──────────────────────────────────────────────────────────

/**
 * MobileNetV2 Inverted Residual block.
 *
 * expand (1×1 → ReLU6) → depthwise 3×3 → project (1×1, linear)
 * Skip connection added when stride === 1 and inChannels === outChannels.
 *
 * @param inChannels   Number of input channels (must match the previous layer).
 * @param outChannels  Number of output channels.
 * @param expandRatio  Channel expansion multiplier before depthwise. Default: 6.
 * @param stride       Depthwise stride. Default: 1.
 *
 * @example
 * new InvertedResidual(32, 16, { expandRatio: 1, stride: 1 })  // first block
 * new InvertedResidual(16, 24, { expandRatio: 6, stride: 2 })  // stride block
 */
export class InvertedResidual extends CompoundLayer {
  readonly name: string;
  private readonly inChannels: number;
  private readonly outChannels: number;
  private readonly expandRatio: number;
  private readonly stride: number;

  constructor(
    inChannels: number,
    outChannels: number,
    opts: {
      expandRatio?: number;
      stride?: number;
      name?: string;
    } = {},
  ) {
    super();
    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.expandRatio = opts.expandRatio ?? 6;
    this.stride = opts.stride ?? 1;
    this.name = opts.name ?? `inv_res_${inChannels}_${outChannels}`;
  }

  build(
    g: Graph,
    input: Tensor,
    inputShape: (number | null)[],
  ): (number | null)[] {
    const hiddenDim = this.inChannels * this.expandRatio;
    const useSkip = this.stride === 1 && this.inChannels === this.outChannels;
    const inputTensor = input;

    // ── Expansion pointwise (skip when expandRatio === 1) ────────────────
    let current: Tensor = input;
    let shape: (number | null)[] = inputShape;

    if (this.expandRatio !== 1) {
      const expand = [
        new Conv2D(hiddenDim, {
          kernelSize: 1,
          useBias: false,
          activation: "linear",
          name: `${this.name}/expand/conv`,
        }),
        new BatchNormalization({ name: `${this.name}/expand/bn` }),
      ];
      for (const l of expand) {
        shape = l.build(g, current, shape);
        current = l.output;
        this.layerParams.push(...l.layerParams);
        this.sublayers.push(l);
      }
      current = relu6(g, current, `${this.name}/expand/relu6`);
    }

    // ── Depthwise 3×3 ───────────────────────────────────────────────────
    const dw = [
      new DepthwiseConv2D({
        kernelSize: 3,
        strides: this.stride,
        padding: "SAME",
        useBias: false,
        name: `${this.name}/dw/conv`,
      }),
      new BatchNormalization({ name: `${this.name}/dw/bn` }),
    ];
    for (const l of dw) {
      shape = l.build(g, current, shape);
      current = l.output;
      this.layerParams.push(...l.layerParams);
      this.sublayers.push(l);
    }
    current = relu6(g, current, `${this.name}/dw/relu6`);

    // ── Pointwise projection (linear — no activation) ────────────────────
    const proj = [
      new Conv2D(this.outChannels, {
        kernelSize: 1,
        useBias: false,
        activation: "linear",
        name: `${this.name}/proj/conv`,
      }),
      new BatchNormalization({ name: `${this.name}/proj/bn` }),
    ];
    for (const l of proj) {
      shape = l.build(g, current, shape);
      current = l.output;
      this.layerParams.push(...l.layerParams);
      this.sublayers.push(l);
    }

    // ── Skip connection ──────────────────────────────────────────────────
    if (useSkip) {
      this.output = add(g, current, inputTensor, `${this.name}/add`);
    } else {
      this.output = current;
    }

    return shape;
  }

  buildFrozen(
    g: Graph,
    input: Tensor,
    inputShape: (number | null)[],
    weights: WeightMap,
  ): { tensor: Tensor; shape: (number | null)[] } {
    const useSkip = this.stride === 1 && this.inChannels === this.outChannels;
    let current: Tensor = input;
    let shape: (number | null)[] = inputShape;
    let idx = 0;

    if (this.expandRatio !== 1) {
      let res = this.sublayers[idx++].buildFrozen(g, current, shape, weights);
      res = this.sublayers[idx++].buildFrozen(
        g,
        res.tensor,
        res.shape,
        weights,
      );
      current = relu6(g, res.tensor, `${this.name}/expand/relu6`);
      shape = res.shape;
    }

    let res = this.sublayers[idx++].buildFrozen(g, current, shape, weights);
    res = this.sublayers[idx++].buildFrozen(g, res.tensor, res.shape, weights);
    current = relu6(g, res.tensor, `${this.name}/dw/relu6`);
    shape = res.shape;

    res = this.sublayers[idx++].buildFrozen(g, current, shape, weights);
    res = this.sublayers[idx++].buildFrozen(g, res.tensor, res.shape, weights);
    current = res.tensor;
    shape = res.shape;

    if (useSkip) {
      current = add(g, current, input, `${this.name}/add`);
    }

    return { tensor: current, shape };
  }

  getConfig(): LayerConfig {
    return {
      type: "InvertedResidual",
      name: this.name,
      inChannels: this.inChannels,
      outChannels: this.outChannels,
      expandRatio: this.expandRatio,
      stride: this.stride,
    } as any;
  }
}

// ── ResidualBlock ─────────────────────────────────────────────────────────────

/**
 * ResNet bottleneck residual block.
 *
 * 1×1 Conv(bottleneckChannels) → BN → ReLU
 * 3×3 Conv(bottleneckChannels) → BN → ReLU
 * 1×1 Conv(outChannels)        → BN
 * + skip projection if inChannels ≠ outChannels
 * → ReLU
 *
 * @param inChannels         Input channel count.
 * @param bottleneckChannels Channels in the 3×3 body. Typically outChannels/4.
 * @param outChannels        Output channel count (= inChannels for identity block).
 * @param stride             Stride of the 3×3 conv. Use 2 for downsampling.
 *
 * @example
 * // Identity block (ResNet res3b)
 * new ResidualBlock(256, 64, 256)
 * // Projection block (ResNet res3a, downsampling)
 * new ResidualBlock(128, 64, 256, { stride: 2 })
 */
export class ResidualBlock extends CompoundLayer {
  readonly name: string;
  private readonly inChannels: number;
  private readonly bottleneckChannels: number;
  private readonly outChannels: number;
  private readonly stride: number;

  constructor(
    inChannels: number,
    bottleneckChannels: number,
    outChannels: number,
    opts: {
      stride?: number;
      name?: string;
    } = {},
  ) {
    super();
    this.inChannels = inChannels;
    this.bottleneckChannels = bottleneckChannels;
    this.outChannels = outChannels;
    this.stride = opts.stride ?? 1;
    this.name = opts.name ?? `res_${inChannels}_${outChannels}`;
  }

  build(
    g: Graph,
    input: Tensor,
    inputShape: (number | null)[],
  ): (number | null)[] {
    const needsProjection =
      this.inChannels !== this.outChannels || this.stride !== 1;
    const inputTensor = input;

    const buildBlock = (
      layers: Layer[],
      start: Tensor,
      startShape: (number | null)[],
    ) => {
      let cur = start;
      let sh = startShape;
      for (const l of layers) {
        sh = l.build(g, cur, sh);
        cur = l.output;
        this.layerParams.push(...l.layerParams);
        this.sublayers.push(l);
      }
      return { cur, sh };
    };

    // ── 1×1 → BN → ReLU ─────────────────────────────────────────────────
    const { cur: h1, sh: sh1 } = buildBlock(
      [
        new Conv2D(this.bottleneckChannels, {
          kernelSize: 1,
          useBias: false,
          activation: "linear",
          name: `${this.name}/1x1a/conv`,
        }),
        new BatchNormalization({ name: `${this.name}/1x1a/bn` }),
      ],
      input,
      inputShape,
    );
    let current = relu(g, h1, `${this.name}/1x1a/relu`);

    // ── 3×3 → BN → ReLU ─────────────────────────────────────────────────
    const { cur: h2, sh: sh2 } = buildBlock(
      [
        new Conv2D(this.bottleneckChannels, {
          kernelSize: 3,
          strides: this.stride,
          padding: "SAME",
          useBias: false,
          activation: "linear",
          name: `${this.name}/3x3/conv`,
        }),
        new BatchNormalization({ name: `${this.name}/3x3/bn` }),
      ],
      current,
      sh1,
    );
    current = relu(g, h2, `${this.name}/3x3/relu`);

    // ── 1×1 → BN (no activation before add) ────────────────────────────
    const { cur: h3, sh: sh3 } = buildBlock(
      [
        new Conv2D(this.outChannels, {
          kernelSize: 1,
          useBias: false,
          activation: "linear",
          name: `${this.name}/1x1b/conv`,
        }),
        new BatchNormalization({ name: `${this.name}/1x1b/bn` }),
      ],
      current,
      sh2,
    );

    // ── Skip / Projection ────────────────────────────────────────────────
    let skip: Tensor;
    if (needsProjection) {
      const { cur: proj, sh: shProj } = buildBlock(
        [
          new Conv2D(this.outChannels, {
            kernelSize: 1,
            strides: this.stride,
            useBias: false,
            activation: "linear",
            name: `${this.name}/proj/conv`,
          }),
          new BatchNormalization({ name: `${this.name}/proj/bn` }),
        ],
        inputTensor,
        inputShape,
      );
      skip = proj;
    } else {
      skip = inputTensor;
    }

    this.output = relu(
      g,
      add(g, h3, skip, `${this.name}/add`),
      `${this.name}/out_relu`,
    );
    return sh3;
  }

  buildFrozen(
    g: Graph,
    input: Tensor,
    inputShape: (number | null)[],
    weights: WeightMap,
  ): { tensor: Tensor; shape: (number | null)[] } {
    const needsProjection =
      this.inChannels !== this.outChannels || this.stride !== 1;

    let current = input;
    let shape = inputShape;
    let idx = 0;

    let res = this.sublayers[idx++].buildFrozen(g, current, shape, weights);
    res = this.sublayers[idx++].buildFrozen(g, res.tensor, res.shape, weights);
    current = relu(g, res.tensor, `${this.name}/1x1a/relu`);
    shape = res.shape;

    res = this.sublayers[idx++].buildFrozen(g, current, shape, weights);
    res = this.sublayers[idx++].buildFrozen(g, res.tensor, res.shape, weights);
    current = relu(g, res.tensor, `${this.name}/3x3/relu`);
    shape = res.shape;

    res = this.sublayers[idx++].buildFrozen(g, current, shape, weights);
    res = this.sublayers[idx++].buildFrozen(g, res.tensor, res.shape, weights);
    let h3 = res.tensor;
    let sh3 = res.shape;

    let skip: Tensor;
    if (needsProjection) {
      let sRes = this.sublayers[idx++].buildFrozen(
        g,
        input,
        inputShape,
        weights,
      );
      sRes = this.sublayers[idx++].buildFrozen(
        g,
        sRes.tensor,
        sRes.shape,
        weights,
      );
      skip = sRes.tensor;
    } else {
      skip = input;
    }

    const out = relu(
      g,
      add(g, h3, skip, `${this.name}/add`),
      `${this.name}/out_relu`,
    );
    return { tensor: out, shape: sh3 };
  }

  getConfig(): LayerConfig {
    return {
      type: "ResidualBlock",
      name: this.name,
      inChannels: this.inChannels,
      bottleneckChannels: this.bottleneckChannels,
      outChannels: this.outChannels,
      stride: this.stride,
    } as any;
  }
}

// ── Re-export primitive layers for convenience under the easy namespace ───────
export {
  Dense,
  Flatten,
  Conv2D,
  DepthwiseConv2D,
  SeparableConv2D,
  MaxPooling2D,
  GlobalAveragePooling2D,
  ZeroPadding2D,
  BatchNormalization,
} from "./layers.js";
