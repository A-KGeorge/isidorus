import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { graph, session, ops, DType } from "../index.js";

// ---------------------------------------------------------------------------
// control_inputs + globalVariablesInitializer
//
// These tests verify:
//   1. addOp() accepts a controlInputs array without throwing
//   2. A NoOp with control deps on assignment ops is valid in TF
//   3. Running the NoOp target via sess.run([],[],[initOp]) causes all
//      control-dep AssignVariableOps to execute — i.e. variables are
//      actually initialised to their specified values
//   4. globalVariablesInitializer correctly wires N init ops into one target
// ---------------------------------------------------------------------------

describe("control_inputs / globalVariablesInitializer", () => {
  it("addOp accepts controlInputs without throwing", () => {
    const g = graph();

    // Build a trivial op to use as a control dependency.
    const noop1 = g.addOp("NoOp", [], {}, "first_noop");
    const noop2 = g.addOp("NoOp", [], {}, "second_noop");

    // This should not throw — the NoOp has two control deps.
    assert.doesNotThrow(() => {
      g.addOp(
        "NoOp",
        [],
        {},
        "controlled_noop",
        ["first_noop", "second_noop"], // controlInputs
      );
    });

    assert.ok(g.hasOp("controlled_noop"));
  });

  it("globalVariablesInitializer creates a NoOp with correct name", () => {
    const g = graph();

    // Two stub "init" NoOps simulating AssignVariableOps.
    g.addOp("NoOp", [], {}, "w/init");
    g.addOp("NoOp", [], {}, "b/init");

    const initAll = ops.globalVariablesInitializer(
      g,
      ["w/init", "b/init"],
      "my_init",
    );

    assert.strictEqual(initAll, "my_init");
    assert.ok(g.hasOp("my_init"));
  });

  it("single variable is initialised and readable after running init target", async () => {
    const g = graph();

    // Create a scalar float32 variable.
    const { handle: wHandle, initOp: wInit } = ops.variableWithInit(
      g,
      [], // scalar shape
      DType.FLOAT32,
      "scalar_w",
      ops.zerosInitializer(g, [], DType.FLOAT32),
    );

    // Override the zero initialiser with a known value for this test.
    // variableWithInit uses zerosInitializer — let's use a constant 42.0
    // by creating our own init op directly.
    const g2 = graph();
    const val42Buf = Buffer.allocUnsafe(4);
    val42Buf.writeFloatLE(42.0, 0);
    const val42 = ops.constant(g2, val42Buf, [], DType.FLOAT32);
    const { handle: wh2, initOp: wi2 } = ops.variableWithInit(
      g2,
      [],
      DType.FLOAT32,
      "w42",
      val42,
    );
    const initAll = ops.globalVariablesInitializer(g2, [wi2]);

    const sess = session(g2);

    // Before init: ReadVariableOp on an uninitialised resource variable
    // throws in TF. We verify init works by running init first, then reading.
    await sess.run([], [], [initAll]);

    const wRead = ops.readVariable(g2, wh2, DType.FLOAT32);
    const [output] = await sess.run([], [wRead]);

    const val = new Float32Array(
      output.data.buffer,
      output.data.byteOffset,
      1,
    )[0];
    assert.strictEqual(val, 42.0);

    sess.destroy();
  });

  it("multiple variables initialised atomically via globalVariablesInitializer", async () => {
    const g = graph();

    // Two variables with distinct initial values.
    const aBuf = Buffer.allocUnsafe(4);
    aBuf.writeFloatLE(1.5, 0);
    const bBuf = Buffer.allocUnsafe(4);
    bBuf.writeFloatLE(-3.0, 0);

    const { handle: aHandle, initOp: aInit } = ops.variableWithInit(
      g,
      [],
      DType.FLOAT32,
      "a",
      ops.constant(g, aBuf, [], DType.FLOAT32),
    );
    const { handle: bHandle, initOp: bInit } = ops.variableWithInit(
      g,
      [],
      DType.FLOAT32,
      "b",
      ops.constant(g, bBuf, [], DType.FLOAT32),
    );

    // Single target to init both.
    const initAll = ops.globalVariablesInitializer(g, [aInit, bInit]);

    const sess = session(g);
    await sess.run([], [], [initAll]);

    const aRead = ops.readVariable(g, aHandle, DType.FLOAT32);
    const bRead = ops.readVariable(g, bHandle, DType.FLOAT32);
    const [aOut, bOut] = await sess.run([], [aRead, bRead]);

    const aVal = new Float32Array(aOut.data.buffer, aOut.data.byteOffset, 1)[0];
    const bVal = new Float32Array(bOut.data.buffer, bOut.data.byteOffset, 1)[0];

    assert.strictEqual(aVal, 1.5);
    assert.strictEqual(bVal, -3.0);

    sess.destroy();
  });

  it("control input op not found throws a clear error", () => {
    const g = graph();
    assert.throws(
      () => g.addOp("NoOp", [], {}, "bad_ctrl", ["nonexistent_op"]),
      /Control input op not found: nonexistent_op/,
    );
  });
});
