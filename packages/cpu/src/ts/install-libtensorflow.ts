/**
 * install-libtensorflow.ts
 *
 * Runtime resolver for libtensorflow.
 *
 * Resolution order:
 *   1. LIBTENSORFLOW_PATH env var     — explicit user or CI override
 *   2. {packageRoot}/libtf/           — downloaded by scripts/install.mjs
 *   3. {packageRoot}/.libtf-config.json — written by install.mjs, stores libtfDir
 *   4. Well-known system paths
 *
 * The interactive prompt that existed in alpha is gone. Installation is now
 * handled entirely by the postinstall script so this module is silent on
 * success and only errors if TF genuinely cannot be found after all fallbacks.
 */

import { existsSync, readFileSync } from "fs";
import { join } from "path";
import { platform } from "os";
import { fileURLToPath } from "url";
import { dirname } from "path";
import { debug, warn } from "./_log.js";

// ── Package root ─────────────────────────────────────────────────────────────

// Works whether running from dist/ (compiled) or src/ts/ (tsx).
const __filename = fileURLToPath(import.meta.url);
const __dirname_ = dirname(__filename);
// dist/       → one level up is packageRoot
// src/ts/     → two levels up is packageRoot
const PACKAGE_ROOT = (() => {
  const candidates = [join(__dirname_, ".."), join(__dirname_, "..", "..")];
  for (const c of candidates) {
    try {
      if (existsSync(join(c, "package.json"))) return c;
    } catch {}
  }
  return candidates[0];
})();

// ── Platform ─────────────────────────────────────────────────────────────────

function libFileName(): string {
  const os = platform();
  if (os === "win32") return "tensorflow.dll";
  if (os === "darwin") return "libtensorflow.dylib";
  return "libtensorflow.so";
}

// ── Detection ─────────────────────────────────────────────────────────────────

function checkDir(dir: string, libFile: string): string | null {
  for (const candidate of [join(dir, "lib", libFile), join(dir, libFile)]) {
    if (existsSync(candidate)) return candidate;
  }
  return null;
}

async function resolveTfPath(): Promise<{
  tfPath: string;
  source: string;
} | null> {
  const libFile = libFileName();

  // 1. LIBTENSORFLOW_PATH env var.
  const envPath = process.env["LIBTENSORFLOW_PATH"];
  if (envPath) {
    const found = checkDir(envPath, libFile);
    if (found) return { tfPath: envPath, source: "LIBTENSORFLOW_PATH" };
    warn(`LIBTENSORFLOW_PATH='${envPath}' set but ${libFile} not found there.`);
    // Don't fall through — if the user set it explicitly, failing loudly is correct.
    return null;
  }

  // 2. Package-local libtf/ (installed by scripts/install.mjs).
  const localLibtf = join(PACKAGE_ROOT, "libtf");
  if (checkDir(localLibtf, libFile)) {
    debug(`Found ${libFile} in ${localLibtf}`);
    return { tfPath: localLibtf, source: "package-local libtf" };
  }

  // 3. .libtf-config.json written by install.mjs (handles unusual package layouts).
  try {
    const configPath = join(PACKAGE_ROOT, ".libtf-config.json");
    if (existsSync(configPath)) {
      const config = JSON.parse(readFileSync(configPath, "utf8")) as {
        libtfDir: string;
      };
      if (config.libtfDir && checkDir(config.libtfDir, libFile)) {
        debug(`Found ${libFile} via .libtf-config.json → ${config.libtfDir}`);
        return { tfPath: config.libtfDir, source: ".libtf-config.json" };
      }
    }
  } catch {}

  // 4. Well-known system paths.
  const systemPaths =
    platform() === "win32"
      ? ["C:\\libtensorflow\\lib", "C:\\Program Files\\libtensorflow\\lib"]
      : platform() === "darwin"
      ? [
          "/usr/local/lib",
          "/opt/homebrew/lib",
          "/usr/local/opt/libtensorflow/lib",
        ]
      : [
          "/usr/local/lib",
          "/usr/lib",
          "/usr/lib/x86_64-linux-gnu",
          "/usr/lib/aarch64-linux-gnu",
        ];

  for (const dir of systemPaths) {
    const found = checkDir(dir, libFile);
    if (found) {
      debug(
        `Found ${libFile} at ${dir} (set LIBTENSORFLOW_PATH=${dir} to silence)`,
      );
      return { tfPath: dir, source: `system path ${dir}` };
    }
  }

  return null;
}

// ── ensureTf ─────────────────────────────────────────────────────────────────

export async function ensureTf(): Promise<string> {
  const resolved = await resolveTfPath();
  if (resolved) {
    process.env["LIBTENSORFLOW_PATH"] = resolved.tfPath;

    // Inject the lib folder into the runtime library search paths so the OS
    // loader can find TensorFlow shared libraries when the .node addon is required.
    // This avoids requiring the user to manually set environment variables.
    const libDir = join(resolved.tfPath, "lib");

    if (platform() === "win32") {
      // Windows: prepend to PATH for tensorflow.dll
      const currentPath = process.env.PATH || "";
      if (!currentPath.toLowerCase().includes(libDir.toLowerCase())) {
        process.env.PATH = `${libDir};${currentPath}`;
      }
    } else if (platform() === "linux") {
      // Linux: prepend to LD_LIBRARY_PATH for libtensorflow.so
      const currentLdPath = process.env.LD_LIBRARY_PATH || "";
      if (!currentLdPath.includes(libDir)) {
        process.env.LD_LIBRARY_PATH = currentLdPath
          ? `${libDir}:${currentLdPath}`
          : libDir;
      }
    } else if (platform() === "darwin") {
      // macOS: prepend to both LD_LIBRARY_PATH and DYLD_LIBRARY_PATH for libtensorflow.dylib
      const currentLdPath = process.env.LD_LIBRARY_PATH || "";
      if (!currentLdPath.includes(libDir)) {
        process.env.LD_LIBRARY_PATH = currentLdPath
          ? `${libDir}:${currentLdPath}`
          : libDir;
      }
      const currentDyldPath = process.env.DYLD_LIBRARY_PATH || "";
      if (!currentDyldPath.includes(libDir)) {
        process.env.DYLD_LIBRARY_PATH = currentDyldPath
          ? `${libDir}:${currentDyldPath}`
          : libDir;
      }
    }

    return resolved.tfPath;
  }

  // Not found anywhere. Give a clear, actionable error.
  const os = platform();
  const isWin = os === "win32";
  const isMac = os === "darwin";

  throw new Error(
    `[isidorus] libtensorflow not found.\n\n` +
      `  The postinstall script (scripts/install.mjs) should have downloaded it\n` +
      `  to ${join(PACKAGE_ROOT, "libtf")}. If that failed, you can:\n\n` +
      `  Option A — re-run the install script:\n` +
      `    node node_modules/@isidorus/cpu/scripts/install.mjs\n\n` +
      `  Option B — install manually and set LIBTENSORFLOW_PATH:\n` +
      (isWin
        ? `    Download: https://storage.googleapis.com/tensorflow/versions/2.18.1/libtensorflow-cpu-windows-x86_64.zip\n` +
          `    Extract to C:\\libtensorflow, add C:\\libtensorflow\\lib to PATH, then:\n` +
          `    set LIBTENSORFLOW_PATH=C:\\libtensorflow\n`
        : isMac
        ? `    brew install libtensorflow  (if available)\n` +
          `    or: wget https://storage.googleapis.com/tensorflow/versions/2.18.1/libtensorflow-cpu-darwin-${
            process.arch === "arm64" ? "arm64" : "x86_64"
          }.tar.gz\n` +
          `        sudo tar -C /usr/local -xzf libtensorflow-cpu-darwin-*.tar.gz\n` +
          `        export LIBTENSORFLOW_PATH=/usr/local\n`
        : `    wget https://storage.googleapis.com/tensorflow/versions/2.18.1/libtensorflow-cpu-linux-${
            process.arch === "arm64" ? "aarch64" : "x86_64"
          }.tar.gz\n` +
          `    sudo tar -C /usr/local -xzf libtensorflow-cpu-linux-*.tar.gz && sudo ldconfig\n` +
          `    export LIBTENSORFLOW_PATH=/usr/local\n`) +
      `\n  Then re-run your application.`,
  );
}
