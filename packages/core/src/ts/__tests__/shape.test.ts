import { test } from "node:test";
import assert from "node:assert";
import {
  rank,
  numElements,
  shapeToTF,
  ShapeFromTF,
  shapesCompatible,
  shapeToString,
} from "../shape.js";

test("Shape utilities", async (t) => {
  await t.test("rank calculates dimensions correctly", () => {
    assert.strictEqual(rank([]), 0);
    assert.strictEqual(rank([4]), 1);
    assert.strictEqual(rank([null, 784]), 2);
  });

  await t.test("numElements handles known and dynamic shapes", () => {
    assert.strictEqual(numElements([]), 1); // Scalar
    assert.strictEqual(numElements([2, 3, 4]), 24);
    assert.strictEqual(numElements([null, 10]), null);
  });

  await t.test("TensorFlow wire format conversion", () => {
    const shape = [null, 224, 224, 3];
    const tfWire = [-1, 224, 224, 3];

    assert.deepStrictEqual(shapeToTF(shape), tfWire);
    assert.deepStrictEqual(ShapeFromTF(tfWire), shape);
  });

  await t.test("shapesCompatible identifies valid matches", () => {
    assert.strictEqual(shapesCompatible([null, 10], [5, 10]), true);
    assert.strictEqual(shapesCompatible([1, 28, 28], [1, 28, 28]), true);
    assert.strictEqual(shapesCompatible([1, 28], [1, 28, 1]), false); // Rank mismatch
    assert.strictEqual(shapesCompatible([5, 10], [6, 10]), false); // Dim mismatch
  });

  await t.test("shapeToString formats correctly", () => {
    assert.strictEqual(shapeToString([]), "()");
    assert.strictEqual(shapeToString([null, 10]), "[?, 10]");
  });
});
