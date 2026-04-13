/**
 * scripts/install.mjs — @isidorus/cpu post-install script.
 *
 * Downloads the prebuilt TensorFlow C library for the current platform and
 * architecture and places it at:
 *
 *   {packageRoot}/libtf/
 *     include/   — headers (needed for compilation from source)
 *     lib/       — shared libraries (tensorflow.dll / .so / .dylib)
 *
 * The prebuilt `.node` files shipped in prebuilds/ have their RPATH set to
 * $ORIGIN/../libtf/lib (Linux) / @loader_path/../libtf/lib (macOS) so the
 * dynamic linker finds libtensorflow without any PATH or LD_LIBRARY_PATH
 * modification required at runtime.
 *
 * On Windows there is no RPATH equivalent. The script writes the lib directory
 * into the user's PATH via `setx` so the DLL is found by the OS loader.
 *
 * Skip conditions (checked in order):
 *   1. SKIP_LIBTF_DOWNLOAD=1         — CI or user explicitly opts out
 *   2. LIBTENSORFLOW_PATH is set      — user is managing TF themselves
 *   3. libtf/lib/{lib} already exists — already installed, nothing to do
 *
 * Override the download URL for air-gapped environments:
 *   LIBTF_DOWNLOAD_URL=https://my-mirror.internal/libtensorflow-cpu-linux-x86_64.tar.gz
 */

import {
  existsSync,
  mkdirSync,
  createWriteStream,
  writeFileSync,
  readdirSync,
  copyFileSync,
} from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { platform, arch } from "node:os";
import { get } from "node:https";
import { exec } from "node:child_process";
import { promisify } from "node:util";
import { tmpdir } from "node:os";

const execAsync = promisify(exec);

// ── Config ────────────────────────────────────────────────────────────────────

const TF_VERSION = "2.18.1";
const TF_BASE_URL = `https://storage.googleapis.com/tensorflow/versions/${TF_VERSION}`;

// Package root is one level up from scripts/.
const __dirname = dirname(fileURLToPath(import.meta.url));
const PACKAGE_DIR = dirname(__dirname);
const LIBTF_DIR = join(PACKAGE_DIR, "libtf");

// ── Platform detection ────────────────────────────────────────────────────────

function getPlatformSpec() {
  const os = platform();
  const cpu = arch();

  if (os === "linux") {
    const archStr = cpu === "arm64" ? "aarch64" : "x86_64";
    return {
      libFile: "libtensorflow.so",
      tarball: `libtensorflow-cpu-linux-${archStr}.tar.gz`,
      extractCmd: (src, dst) => `tar -C '${dst}' -xzf '${src}'`,
      postExtract: async () => {
        // Refresh the dynamic linker cache if ldconfig is available.
        // This is best-effort — failure doesn't break the install.
        try {
          await execAsync("ldconfig " + join(LIBTF_DIR, "lib"));
        } catch {}
      },
      winPath: null,
    };
  }

  if (os === "darwin") {
    const archStr = cpu === "arm64" ? "arm64" : "x86_64";
    return {
      libFile: "libtensorflow.dylib",
      tarball: `libtensorflow-cpu-darwin-${archStr}.tar.gz`,
      extractCmd: (src, dst) => `tar -C '${dst}' -xzf '${src}'`,
      postExtract: async () => {},
      winPath: null,
    };
  }

  if (os === "win32") {
    return {
      libFile: "tensorflow.dll",
      tarball: "libtensorflow-cpu-windows-x86_64.zip",
      extractCmd: (src, dst) =>
        `powershell -NoProfile -Command "Expand-Archive -Path '${src}' -DestinationPath '${dst}' -Force"`,
      postExtract: async () => {
        // No permanent PATH modifications required.
        // install-libtensorflow.ts handles injecting the path into process.env.PATH
        // at runtime to resolve tensorflow.dll without requiring a terminal restart.
      },
      winPath: join(LIBTF_DIR, "lib"),
    };
  }

  throw new Error(`[isidorus] Unsupported platform: ${os}`);
}

// ── Skip conditions ───────────────────────────────────────────────────────────

function shouldSkip(spec) {
  // Explicit opt-out.
  if (process.env.SKIP_LIBTF_DOWNLOAD === "1") {
    console.log(
      "[isidorus] SKIP_LIBTF_DOWNLOAD=1 — skipping libtensorflow download.",
    );
    return true;
  }

  // User is managing TF themselves via LIBTENSORFLOW_PATH.
  if (process.env.LIBTENSORFLOW_PATH) {
    console.log(
      `[isidorus] LIBTENSORFLOW_PATH is set — skipping download (${process.env.LIBTENSORFLOW_PATH}).`,
    );
    return true;
  }

  // Already installed.
  const libPath = join(LIBTF_DIR, "lib", spec.libFile);
  if (existsSync(libPath)) {
    console.log(`[isidorus] libtensorflow already installed at ${LIBTF_DIR}.`);
    return true;
  }

  return false;
}

// ── Library copying ───────────────────────────────────────────────────────────

/**
 * Copy shared libraries to prebuilds/{platform}-{arch} so the prebuilt .node
 * file can find them via RPATH ($ORIGIN on Linux, @loader_path on macOS).
 *
 * This is especially important on Linux where the .node file's RUNPATH includes
 * $ORIGIN, allowing the dynamic linker to find libtensorflow.so without modifying
 * LD_LIBRARY_PATH or PATH.
 *
 * On Windows, this is not needed since the DLL is found via PATH.
 */
function copyLibrariesToPrebuilds(spec) {
  const os = platform();
  if (os === "win32") {
    return; // Windows uses PATH, not needed
  }

  const libDir = join(LIBTF_DIR, "lib");
  if (!existsSync(libDir)) {
    return; // Nothing to copy if lib dir doesn't exist
  }

  let libFilePatterns = [];
  if (os === "linux") {
    libFilePatterns = [".so", ".so.2", ".so.2.18.1"];
  } else if (os === "darwin") {
    libFilePatterns = [".dylib", ".dylib.2", ".dylib.2.18.1"];
  } else {
    return;
  }

  const cpu = arch();
  const archStr = cpu === "arm64" ? "arm64" : "x86_64";
  const platformStr =
    os === "darwin" ? `darwin-${archStr}` : `linux-${archStr}`;
  const prebuildsDir = join(PACKAGE_DIR, "prebuilds", platformStr);

  mkdirSync(prebuildsDir, { recursive: true });

  try {
    const files = readdirSync(libDir);
    let copiedCount = 0;

    for (const file of files) {
      // Match files ending with our library extensions
      if (
        libFilePatterns.some((pattern) => file.includes(pattern)) &&
        file.startsWith("libtensorflow")
      ) {
        const src = join(libDir, file);
        const dst = join(prebuildsDir, file);
        try {
          copyFileSync(src, dst);
          copiedCount++;
        } catch (e) {
          // Non-fatal — log but continue
          console.warn(`[isidorus] failed to copy ${file}: ${e.message}`);
        }
      }
    }

    if (copiedCount > 0) {
      console.log(
        `  Copied ${copiedCount} library files to prebuilds/${platformStr}/`,
      );
    }
  } catch (e) {
    // Non-fatal — prebuilt might already have them or user is using RPATH
    console.warn(`[isidorus] library copy warning: ${e.message}`);
  }
}

// ── Download ──────────────────────────────────────────────────────────────────

function downloadFile(url, dest) {
  return new Promise((resolve, reject) => {
    console.log(`  Downloading: ${url}`);
    const file = createWriteStream(dest);

    const request = (u) =>
      get(u, (res) => {
        // Follow redirects.
        if (res.statusCode === 301 || res.statusCode === 302) {
          request(res.headers.location);
          return;
        }
        if (res.statusCode !== 200) {
          reject(new Error(`HTTP ${res.statusCode} from ${u}`));
          return;
        }

        const total = parseInt(res.headers["content-length"] ?? "0", 10);
        let downloaded = 0;
        let lastPct = -1;

        res.on("data", (chunk) => {
          downloaded += chunk.length;
          if (total > 0) {
            const pct = Math.floor((downloaded / total) * 100);
            if (pct !== lastPct && pct % 5 === 0) {
              process.stdout.write(
                `\r  Progress: ${pct}% (${(downloaded / 1024 / 1024).toFixed(
                  1,
                )} MB)`,
              );
              lastPct = pct;
            }
          }
        });

        res.pipe(file);
        file.on("finish", () => {
          process.stdout.write("\n");
          file.close();
          resolve();
        });
      });

    request(url);
    file.on("error", reject);
  });
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  let spec;
  try {
    spec = getPlatformSpec();
  } catch (e) {
    console.warn(`[isidorus] ${e.message}`);
    console.warn(
      "[isidorus] Set LIBTENSORFLOW_PATH to your TF install directory and rebuild.",
    );
    return;
  }

  if (shouldSkip(spec)) return;

  const downloadUrl =
    process.env.LIBTF_DOWNLOAD_URL ?? `${TF_BASE_URL}/${spec.tarball}`;
  const tmpFile = join(tmpdir(), spec.tarball);

  console.log(`\n[isidorus] Installing libtensorflow ${TF_VERSION}...`);
  console.log(`  Platform : ${platform()}-${arch()}`);
  console.log(`  Tarball  : ${spec.tarball}`);
  console.log(`  Dest     : ${LIBTF_DIR}\n`);

  mkdirSync(LIBTF_DIR, { recursive: true });

  try {
    await downloadFile(downloadUrl, tmpFile);
  } catch (e) {
    console.error(`\n[isidorus] Download failed: ${e.message}`);
    console.error(
      `[isidorus] You can download manually and set LIBTENSORFLOW_PATH:`,
    );
    console.error(`  ${downloadUrl}`);
    // Non-fatal — user can install manually.
    return;
  }

  console.log(`  Extracting to ${LIBTF_DIR}...`);
  try {
    const cmd = spec.extractCmd(tmpFile, LIBTF_DIR);
    await execAsync(cmd);
  } catch (e) {
    console.error(`[isidorus] Extraction failed: ${e.message}`);
    console.error(
      `[isidorus] Archive saved at ${tmpFile} — extract manually to ${LIBTF_DIR}.`,
    );
    return;
  }

  // Platform-specific post-install (ldconfig, PATH modification, etc.)
  await spec.postExtract();

  // Copy shared libraries to prebuilds dir for RPATH lookup on Linux/macOS.
  copyLibrariesToPrebuilds(spec);

  // Write a config file so the runtime resolver can find TF even without
  // LIBTENSORFLOW_PATH being set in the environment.
  const configPath = join(PACKAGE_DIR, ".libtf-config.json");
  writeFileSync(
    configPath,
    JSON.stringify(
      {
        version: TF_VERSION,
        installedAt: new Date().toISOString(),
        libtfDir: LIBTF_DIR,
        platform: platform(),
        arch: arch(),
      },
      null,
      2,
    ),
  );

  const libPath = join(LIBTF_DIR, "lib", spec.libFile);
  if (!existsSync(libPath)) {
    console.error(`[isidorus] ⚠  ${spec.libFile} not found after extraction.`);
    console.error(`[isidorus]    Expected: ${libPath}`);
    console.error(
      `[isidorus]    The archive structure may have changed — check ${LIBTF_DIR}.`,
    );
    return;
  }

  console.log(
    `\n[isidorus] libtensorflow ${TF_VERSION} installed successfully.`,
  );
  console.log(`  Library  : ${libPath}`);
  if (platform() !== "win32") {
    console.log(
      `\n  Prebuilt .node files resolve the library automatically via RPATH.`,
    );
    console.log(
      `  If compiling from source, set:\n    LIBTENSORFLOW_PATH=${LIBTF_DIR}`,
    );
  }
  console.log();
}

main().catch((e) => {
  // Never throw from postinstall — a failed optional download should not
  // break `npm install` for the parent project.
  console.warn(`[isidorus] post-install warning: ${e.message}`);
});
