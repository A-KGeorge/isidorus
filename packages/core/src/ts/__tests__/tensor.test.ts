import { test } from "node:test";
import assert from "node:assert";
import { makeTensor, tensorId } from "../tensor.js";
import { DType } from "../dtype.js";

test("Tensor descriptors", async (t) => {
  await t.test("makeTensor assigns correct properties", () => {
    const t = makeTensor("MatMul", 0, DType.FLOAT32, [2, 2]);
    assert.strictEqual(t.opName, "MatMul");
    assert.strictEqual(t.index, 0);
    assert.strictEqual(t.dtype, DType.FLOAT32);
    assert.deepStrictEqual(t.shape, [2, 2]);
    assert.strictEqual(t.name, "MatMul"); // Default name
  });

  await t.test("makeTensor uses custom name if provided", () => {
    const t = makeTensor("Const", 0, DType.INT32, [], "my_const");
    assert.strictEqual(t.name, "my_const");
  });

  await t.test("tensorId generates valid wire keys", () => {
    const t = makeTensor("Add", 1, null, null);
    assert.strictEqual(tensorId(t), "Add:1");
  });
});
