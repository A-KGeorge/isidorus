/**
 * DType - element type enum.
 *
 * Values match TF_DataType from the Tensorflow C API exactly so they
 * can be passed across the N-API boundary without conversion.
 *
 * Also matches the jude-map DType enum, for the subset of types that
 * jude-map supports (FLOAT32=1 in TF, 0 in jude-map - translation
 * happens in @isidorus/cpu's native layer)
 */
export enum DType {
  FLOAT32 = 1, // TF_FLOAT
  FLOAT64 = 2, // TF_DOUBLE
  INT32 = 3, // TF_INT32
  UINT8 = 4, // TF_UINT8
  INT16 = 5, // TF_INT16
  INT8 = 6, // TF_INT8
  STRING = 7, // TF_STRING
  INT64 = 9, // TF_INT64
  BOOL = 10, // TF_BOOL
  UINT16 = 17, // TF_UINT16
  UINT32 = 22, // TF_UINT32
  UINT64 = 23, // TF_UINT64
}

/** Byte size of one element for numeric dtypes */
export function dtypeItemSize(dtype: DType): number {
  switch (dtype) {
    case DType.FLOAT32:
      return 4;
    case DType.FLOAT64:
      return 8;
    case DType.INT32:
      return 4;
    case DType.UINT8:
      return 1;
    case DType.INT8:
      return 1;
    case DType.INT16:
      return 2;
    case DType.UINT16:
      return 2;
    case DType.INT64:
      return 8;
    case DType.UINT64:
      return 8;
    case DType.UINT32:
      return 4;
    case DType.BOOL:
      return 1;
    default:
      throw new Error(`No itemsize for DType ${dtype}`);
  }
}

/** Human-readable name for a DType */
export function dtypeName(dtype: DType): string {
  return DType[dtype] ?? `DType(${dtype})`;
}
