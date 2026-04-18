/**
 * toon.ts — Token Oriented Object Notation
 *
 * A compact binary serialization format for isidorus model manifests.
 * Designed for two properties JSON lacks:
 *   1. Compact — no repeated key strings, integers in varint encoding
 *   2. Fast to parse — single forward scan, no backtracking, no regex
 *
 * Format overview
 * ───────────────
 * All multi-byte integers are little-endian.
 * Strings are length-prefixed: [varint byte_length][utf8 bytes].
 * The file begins with a 4-byte magic header: 0x544F4F4E ("TOON").
 *
 * Token types (1 byte tag):
 *   0x01  NULL
 *   0x02  TRUE
 *   0x03  FALSE
 *   0x04  INT      [int32 LE]            4 bytes
 *   0x05  FLOAT    [float64 LE]          8 bytes
 *   0x06  STRING   [varint len][utf8]
 *   0x07  BYTES    [varint len][bytes]   raw binary blob
 *   0x08  ARRAY    [varint count] items
 *   0x09  OBJECT   [varint count] key-value pairs: STRING value STRING value...
 *                  (keys are bare string bodies — no tag byte, length-prefixed)
 *   0x0A  BIGINT   [varint len][LE bytes] arbitrary precision, unsigned
 *
 * Varints: unsigned LEB128 encoding (same as protobuf).
 *
 * Usage
 * ─────
 *   import { toonEncode, toonDecode } from "./toon.js";
 *   const bytes = toonEncode(manifest);
 *   const back  = toonDecode(bytes);
 */

// ── Magic ─────────────────────────────────────────────────────────────────────

const MAGIC = 0x4e4f4f54; // "TOON" in little-endian uint32

// ── Tags ──────────────────────────────────────────────────────────────────────

const T_NULL = 0x01;
const T_TRUE = 0x02;
const T_FALSE = 0x03;
const T_INT = 0x04;
const T_FLOAT = 0x05;
const T_STRING = 0x06;
const T_BYTES = 0x07;
const T_ARRAY = 0x08;
const T_OBJECT = 0x09;

const MAX_VALUE_BYTE_LENGTH = 256 * 1024 * 1024; // 256 MB

// ── LEB128 varint ─────────────────────────────────────────────────────────────

function varintSize(n: number): number {
  let size = 0;
  do {
    n >>>= 7;
    size++;
  } while (n);
  return size;
}

function writeVarint(buf: Buffer, offset: number, n: number): number {
  while (n > 0x7f) {
    buf[offset++] = (n & 0x7f) | 0x80;
    n >>>= 7;
  }
  buf[offset++] = n & 0x7f;
  return offset;
}

function readVarint(
  buf: Buffer,
  offset: number,
  checkLength = false,
): { value: number; offset: number } {
  let result = 0;
  let shift = 0;
  while (true) {
    if (offset >= buf.length)
      throw new RangeError("TOON: varint read past end of buffer");
    const byte = buf[offset++];
    result |= (byte & 0x7f) << shift;
    if (!(byte & 0x80)) break;
    shift += 7;
    if (shift > 28) throw new RangeError("TOON: varint overflow (> 28 bits)");
  }
  // Fix 8: coerce to unsigned 32-bit and verify it is a safe non-negative int.
  const value = result >>> 0;
  if (!Number.isSafeInteger(value) || value < 0)
    throw new RangeError(`TOON: varint decoded to invalid value ${value}`);
  // Fix 8: reject implausible lengths before any Buffer allocation.
  if (checkLength && value > MAX_VALUE_BYTE_LENGTH)
    throw new RangeError(
      `TOON: value byte-length ${value} exceeds maximum ${MAX_VALUE_BYTE_LENGTH}`,
    );
  return { value, offset };
}

// ── Encoder ───────────────────────────────────────────────────────────────────

/** Compute the encoded byte size of a value without allocating. */
function sizeOf(value: unknown): number {
  if (value === null || value === undefined) return 1; // tag only
  if (value === true || value === false) return 1; // tag only
  if (typeof value === "number") {
    if (Number.isInteger(value) && value >= -2147483648 && value <= 2147483647)
      return 1 + 4; // tag + int32
    return 1 + 8; // tag + float64
  }
  if (typeof value === "string") {
    const len = Buffer.byteLength(value, "utf8");
    return 1 + varintSize(len) + len; // tag + vlen + body
  }
  if (Buffer.isBuffer(value)) {
    return 1 + varintSize(value.byteLength) + value.byteLength; // tag + vlen + body
  }
  if (Array.isArray(value)) {
    return (
      1 +
      varintSize(value.length) +
      value.reduce((s: number, v: unknown) => s + sizeOf(v), 0)
    );
  }
  if (typeof value === "object" && value !== null) {
    const keys = Object.keys(value as object);
    return (
      1 +
      varintSize(keys.length) +
      keys.reduce((s, k) => {
        const klen = Buffer.byteLength(k, "utf8");
        return s + varintSize(klen) + klen + sizeOf((value as any)[k]);
      }, 0)
    );
  }
  // Fallback: serialise as string representation.
  const str = String(value);
  const len = Buffer.byteLength(str, "utf8");
  return 1 + varintSize(len) + len;
}

function encodeInto(buf: Buffer, offset: number, value: unknown): number {
  if (value === null || value === undefined) {
    buf[offset++] = T_NULL;
    return offset;
  }
  if (value === true) {
    buf[offset++] = T_TRUE;
    return offset;
  }
  if (value === false) {
    buf[offset++] = T_FALSE;
    return offset;
  }
  if (typeof value === "number") {
    if (
      Number.isInteger(value) &&
      value >= -2147483648 &&
      value <= 2147483647
    ) {
      buf[offset++] = T_INT;
      buf.writeInt32LE(value, offset);
      return offset + 4;
    }
    buf[offset++] = T_FLOAT;
    buf.writeDoubleLike(value, offset);
    return offset + 8;
  }
  if (typeof value === "string") {
    buf[offset++] = T_STRING;
    const len = Buffer.byteLength(value, "utf8");
    offset = writeVarint(buf, offset, len);
    buf.write(value, offset, "utf8");
    return offset + len;
  }
  if (Buffer.isBuffer(value)) {
    buf[offset++] = T_BYTES;
    offset = writeVarint(buf, offset, value.byteLength);
    value.copy(buf, offset);
    return offset + value.byteLength;
  }
  if (Array.isArray(value)) {
    buf[offset++] = T_ARRAY;
    offset = writeVarint(buf, offset, value.length);
    for (const item of value) offset = encodeInto(buf, offset, item);
    return offset;
  }
  if (typeof value === "object") {
    const keys = Object.keys(value as object);
    buf[offset++] = T_OBJECT;
    offset = writeVarint(buf, offset, keys.length);
    for (const k of keys) {
      const klen = Buffer.byteLength(k, "utf8");
      offset = writeVarint(buf, offset, klen);
      buf.write(k, offset, "utf8");
      offset += klen;
      offset = encodeInto(buf, offset, (value as any)[k]);
    }
    return offset;
  }
  // Fallback → string
  const str = String(value);
  buf[offset++] = T_STRING;
  const len = Buffer.byteLength(str, "utf8");
  offset = writeVarint(buf, offset, len);
  buf.write(str, offset, "utf8");
  return offset + len;
}

/**
 * toonEncode — serialise any JSON-compatible value to a TOON Buffer.
 * Supports all JSON primitives plus `Buffer` values (serialised as T_BYTES).
 */
export function toonEncode(value: unknown): Buffer {
  const bodySize = sizeOf(value);
  const buf = Buffer.allocUnsafe(4 + bodySize); // 4-byte magic
  buf.writeUInt32LE(MAGIC, 0);
  encodeInto(buf, 4, value);
  return buf;
}

// ── Decoder ───────────────────────────────────────────────────────────────────

/** Read a key string (no tag byte — keys are bare length-prefixed utf8). */
const BLOCKED_KEYS = new Set(["__proto__", "constructor", "prototype"]);

function readKey(buf: Buffer, offset: number): { key: string; offset: number } {
  // Fix 8: pass checkLength=true so oversized key lengths are rejected.
  const { value: len, offset: o1 } = readVarint(buf, offset, true);
  if (o1 + len > buf.length)
    throw new RangeError("TOON: key string extends past end of buffer");
  const key = buf.toString("utf8", o1, o1 + len);
  return { key, offset: o1 + len };
}

function nullProtoToPlain(obj: Record<string | symbol, unknown>): object {
  const plain = {};
  for (const key of Object.getOwnPropertyNames(obj)) {
    Object.defineProperty(
      plain,
      key,
      Object.getOwnPropertyDescriptor(obj, key)!,
    );
  }
  return plain;
}

function decodeValue(
  buf: Buffer,
  offset: number,
): { value: unknown; offset: number } {
  const tag = buf[offset++];
  switch (tag) {
    case T_NULL:
      return { value: null, offset };
    case T_TRUE:
      return { value: true, offset };
    case T_FALSE:
      return { value: false, offset };
    case T_INT: {
      const v = buf.readInt32LE(offset);
      return { value: v, offset: offset + 4 };
    }
    case T_FLOAT: {
      const v = buf.readDoubleLike(offset);
      return { value: v, offset: offset + 8 };
    }
    case T_STRING: {
      const { value: len, offset: o1 } = readVarint(buf, offset);
      const str = buf.toString("utf8", o1, o1 + len);
      return { value: str, offset: o1 + len };
    }
    case T_BYTES: {
      const { value: len, offset: o1 } = readVarint(buf, offset);
      const bytes = buf.slice(o1, o1 + len);
      return { value: bytes, offset: o1 + len };
    }
    case T_ARRAY: {
      const { value: count, offset: o1 } = readVarint(buf, offset);
      const arr: unknown[] = [];
      let o = o1;
      for (let i = 0; i < count; i++) {
        const r = decodeValue(buf, o);
        arr.push(r.value);
        o = r.offset;
      }
      return { value: arr, offset: o };
    }
    case T_OBJECT: {
      const { value: count, offset: o1 } = readVarint(buf, offset);
      const obj = Object.create(null) as Record<string, unknown>;
      let o = o1;
      for (let i = 0; i < count; i++) {
        const kr = readKey(buf, o);
        const vr = decodeValue(buf, kr.offset);

        if (!BLOCKED_KEYS.has(kr.key)) {
          obj[kr.key] = vr.value;
        }
        o = vr.offset;
      }

      return { value: nullProtoToPlain(obj), offset: o };
    }
    default:
      throw new RangeError(
        `TOON: unknown tag 0x${tag.toString(16)} at offset ${offset - 1}`,
      );
  }
}

/**
 * toonDecode — deserialise a TOON Buffer back to a value.
 * Buffer (T_BYTES) values are returned as Node.js Buffer instances.
 * @throws RangeError on malformed input.
 */
export function toonDecode(buf: Buffer): unknown {
  if (buf.byteLength < 4) throw new RangeError("TOON: buffer too short");
  const magic = buf.readUInt32LE(0);
  if (magic !== MAGIC)
    throw new RangeError(`TOON: bad magic 0x${magic.toString(16)}`);
  const { value } = decodeValue(buf, 4);
  return value;
}

// ── Buffer readDoubleLike / writeDoubleLike polyfill ──────────────────────────
// Node.js Buffer has readDoubleLE / writeDoubleLE but not the generic form.
// Attach as prototype extensions if missing (they are present in all modern Node).

declare global {
  interface Buffer {
    readDoubleLike(offset: number): number;
    writeDoubleLike(value: number, offset: number): void;
  }
}

if (!Buffer.prototype.readDoubleLike) {
  Buffer.prototype.readDoubleLike = function (o: number) {
    return this.readDoubleLE(o);
  };
  Buffer.prototype.writeDoubleLike = function (v: number, o: number) {
    this.writeDoubleLE(v, o);
  };
}
