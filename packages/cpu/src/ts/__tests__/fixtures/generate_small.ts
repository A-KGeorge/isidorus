import { DType } from "@isidorus/core";
import {
  Graph,
  getAddon,
  constGlorot,
  constZeros,
  convBnRelu6,
  invertedResidual,
  globalAvgPool,
  softmax,
  matmul,
  biasAdd,
} from "../../index.js";

export function buildSmallGraph(): Graph {
  const g = new Graph(new (getAddon().Graph)());
  const [x] = g.addOp(
    "Placeholder",
    [],
    {
      dtype: { kind: "type", value: DType.FLOAT32 },
      shape: { kind: "shape", value: [1, 224, 224, 3] },
    },
    "inputs",
  );

  const invResTable: [number, number, number, number][] = [
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
  ];

  let [h, c] = convBnRelu6(g, x, 3, 32, 3, 2, "conv_stem");

  let blockIdx = 0;
  for (const [t, outC, n, s] of invResTable) {
    for (let i = 0; i < n; i++) {
      const stride = i === 0 ? s : 1;
      const name = `ir_${blockIdx}`;
      [h, c] = invertedResidual(g, h, c, outC, stride, t, name);
      blockIdx++;
    }
  }

  [h, c] = convBnRelu6(g, h, 320, 1280, 1, 1, "conv_head");
  h = globalAvgPool(g, h, "avg_pool");

  const wOut = constGlorot(g, [1280, 1000], "classifier/w");
  const bOut = constZeros(g, 1000, "classifier/b");
  const logits = biasAdd(
    g,
    matmul(g, h, wOut, {}, "classifier/mm"),
    bOut,
    "classifier/ba",
  );
  const out = softmax(g, logits, "predictions");
  g.addOp("Identity", [out], {}, "output_identity");

  return g;
}
