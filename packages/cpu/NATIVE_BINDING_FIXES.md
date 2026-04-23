# @isidorus/cpu - Native Binding Fixes

## Quick Fix Summary

The native TensorFlow bindings now load automatically without requiring manual environment variable configuration. Developers no longer need to set `LIBTENSORFLOW_PATH` or `LD_LIBRARY_PATH` manually.

## What Was Fixed

### Problem

- After `npm install`, developers were required to run code with custom environment variables:
  ```bash
  LD_LIBRARY_PATH=/path/to/tf/lib LIBTENSORFLOW_PATH=/path/to/tf node app.js
  ```
- Omitting these variables resulted in cryptic symbol lookup errors:
  ```
  node: symbol lookup error: .../prebuilds/linux-x64/@isidorus+cpu.node: undefined symbol: TF_NewGraph
  ```

### Root Cause

The postinstall script was:

1. Downloading TensorFlow C libraries
2. Moving them to `prebuilds/{platform}/` for the `.node` binary
3. Creating symlinks from `libtf/lib` back to those files
4. **Deleting the `libtf/lib` directory**, breaking the symlinks
5. The `.node` binary couldn't find the required TensorFlow symbols

### Solution

1. **Keep symlinks intact** - The postinstall script no longer deletes `libtf/lib` on Linux
2. **Smart library detection** - Runtime loader checks:
   - Prebuilds directory first (where `node-gyp-build` finds prebuilt binaries)
   - `libtf/lib` directory with symlinks
   - System library paths
   - Explicit `LIBTENSORFLOW_PATH` environment variable
3. **Automatic environment setup** - LD_LIBRARY_PATH is automatically configured before loading the native module
4. **Better rpath support** - Build configuration includes prebuilds directory in runtime search paths

## How It Works Now

```javascript
// Simply import - no environment setup needed!
import cpu from "@isidorus/cpu";

const graph = cpu.graph();
// ... use the module normally
```

The package initialization:

1. Calls `ensureTf()` which automatically finds and sets up TensorFlow
2. Sets appropriate environment variables (LD_LIBRARY_PATH, DYLD_LIBRARY_PATH, PATH)
3. Loads the native `.node` module
4. Everything works without user intervention

## Advanced Usage

### Using Custom TensorFlow

If you have TensorFlow installed elsewhere:

```bash
export LIBTENSORFLOW_PATH=/opt/tensorflow
npm install @isidorus/cpu
```

Or pass it during install:

```bash
LIBTENSORFLOW_PATH=/opt/tensorflow npm install @isidorus/cpu
```

### Troubleshooting

If you still encounter issues:

```bash
# Re-run the postinstall script
node node_modules/@isidorus/cpu/scripts/postinstall.mjs

# Or skip TensorFlow download and provide your own
LIBTENSORFLOW_PATH=/path/to/tf npm install @isidorus/cpu

# Or skip TensorFlow download entirely (if you're handling it separately)
SKIP_LIBTF_DOWNLOAD=1 npm install @isidorus/cpu
```

## Files Modified

- **scripts/postinstall.mjs** - Fixed cleanup to preserve symlinks on Linux
- **src/ts/install-libtensorflow.ts** - Enhanced library resolution and environment setup
- **binding.gyp** - Added better rpath support for library discovery

## Testing

The package includes tests to verify library resolution works correctly:

```bash
npm test
```

Specifically, run the new linking tests:

```bash
npm test -- libtf-resolution
```

## Platform Support

- ✅ **Linux x64** - Fully tested and working
- ✅ **macOS ARM64** - Uses dylib loading
- ✅ **Windows x64** - Uses DLL loading
- ✅ **Linux aarch64** - Supported via fallback to system libraries
- ⚠️ **Other platforms** - May require manual LIBTENSORFLOW_PATH setup

## Performance Impact

No performance impact. The library resolution happens once at module load time and is very fast (usually < 1ms).

## Breaking Changes

None - this is a transparent improvement that fixes the loading process without changing the public API.
