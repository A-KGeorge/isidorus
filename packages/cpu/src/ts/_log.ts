/**
 * Internal logging utilities for @isidorus/cpu.
 *
 * debug() — gated behind ISIDORUS_DEBUG=1. Use for diagnostic output that
 *           helps developers understand what the library is doing internally
 *           (thread config, autotuner results, shape resolution, etc.).
 *           Silent in production by default.
 *
 * warn()  — always prints to stderr. Use for actionable warnings that the
 *           user should know about regardless of debug settings
 *           (thread over-subscription, missing env vars, etc.).
 */

const DEBUG = process.env["ISIDORUS_DEBUG"] === "1";

/** Prints to stderr only when ISIDORUS_DEBUG=1. */
export function debug(msg: string): void {
  if (DEBUG) process.stderr.write(`[isidorus] ${msg}\n`);
}

/** Always prints to stderr — for actionable warnings. */
export function warn(msg: string): void {
  process.stderr.write(`[isidorus] warning: ${msg}\n`);
}
