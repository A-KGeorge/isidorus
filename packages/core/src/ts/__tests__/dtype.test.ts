import { test } from "node:test";
import assert from "node:assert";
import { DType, dtypeItemSize, dtypeName } from "../dtype.js";

test("DType utilities", async (t) => {
  await t.test("dtypeItemSize returns correct byte sizes", () => {
    assert.strictEqual(dtypeItemSize(DType.FLOAT32), 4);
    assert.strictEqual(dtypeItemSize(DType.UINT8), 1);
    assert.strictEqual(dtypeItemSize(DType.FLOAT64), 8);
    assert.strictEqual(dtypeItemSize(DType.INT64), 8);
    assert.strictEqual(dtypeItemSize(DType.BOOL), 1);
  });

  await t.test("dtypeItemSize throws for unknown types", () => {
    assert.throws(() => dtypeItemSize(999 as DType), /No itemsize for DType/);
  });

  await t.test("dtypeName returns string representation", () => {
    assert.strictEqual(dtypeName(DType.FLOAT32), "FLOAT32");
    assert.strictEqual(dtypeName(DType.INT32), "INT32");
    assert.strictEqual(dtypeName(12345 as DType), "DType(12345)");
  });
});
