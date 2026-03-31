/**
 * _native.ts — internal addon reference.
 *
 * Holds the loaded native addon so internal modules (inference-pool, etc.)
 * can access it without importing from the package's own entry point
 * (@isidorus/cpu), which would be a circular dependency.
 *
 * index.ts calls setAddon() once after node-gyp-build resolves.
 * Any internal module that needs Graph or Session construction calls getAddon().
 */

let _addon: any = null;

export function setAddon(addon: any): void {
  _addon = addon;
}

export function getAddon(): any {
  if (!_addon) {
    throw new Error(
      "[isidorus] Native addon not initialised. " +
        "Ensure @isidorus/cpu is imported before calling InferencePool.create().",
    );
  }
  return _addon;
}
