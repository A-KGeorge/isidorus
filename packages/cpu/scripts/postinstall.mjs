/**
 * scripts/install.mjs — @isidorus/cpu post-install script.
 *
 * Downloads the prebuilt TensorFlow C library for the current platform and
 * places it at:
 *
 *   {packageRoot}/libtf/
 *     include/   — headers (needed for compilation from source)
 *     lib/       — shared libraries (libtensorflow.so / .dylib / tensorflow.dll)
 *
 * ── Build variant selection (Linux x86_64 only) ────────────────────────────
 *
 * On Linux x86_64 we ship custom MKL-linked builds from the isidorus GitHub
 * releases, which significantly outperform the official CPU build on modern
 * hardware (oneDNN + MKL-DNN kernel selection, AVX2/FMA code paths).
 *
 * Variants (in priority order):
 *   mkl-avx2   MKL + AVX2 + FMA — best performance, requires Haswell (2013+)
 *   mkl        MKL only — compatible with any x86_64; still faster than vanilla
 *   cpu        Official libtensorflow.so from storage.googleapis.com (fallback)
 *
 * Variant selection:
 *   1. LIBTF_VARIANT env var   — force a specific variant (mkl-avx2 | mkl | cpu)
 *   2. /proc/cpuinfo flags     — auto-detect avx2 + fma → mkl-avx2, else mkl
 *   3. GitHub download fails   — falls back to official TF automatically
 *
 * Other platforms (arm64 Linux, macOS, Windows) always use the official build
 * and skip the variant logic entirely.
 *
 * ── RPATH ─────────────────────────────────────────────────────────────────
 *
 * Prebuilt .node files have RPATH set to:
 *   Linux:  $ORIGIN/../libtf/lib
 *   macOS:  @loader_path/../libtf/lib
 *
 * No LD_LIBRARY_PATH or PATH modification is required at runtime.
 * Windows uses PATH injection in install-libtensorflow.ts instead.
 *
 * ── Skip conditions ────────────────────────────────────────────────────────
 *   SKIP_LIBTF_DOWNLOAD=1       — CI or explicit opt-out
 *   LIBTENSORFLOW_PATH is set   — user manages TF themselves
 *   libtf/lib/{lib} exists      — already installed
 *
 * ── Air-gap / mirror override ──────────────────────────────────────────────
 *   LIBTF_DOWNLOAD_URL=https://mirror/libtensorflow-....tar.gz
 *   (overrides both the variant URL and the official URL fallback)
 */

import {
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
  readdirSync,
  renameSync,
  symlinkSync,
  unlinkSync,
  createWriteStream,
  cpSync,
} from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { platform, arch, tmpdir } from "node:os";
import { get } from "node:https";
import { exec } from "node:child_process";
import { promisify } from "node:util";

const execAsync = promisify(exec);

// ── Config ────────────────────────────────────────────────────────────────────

const TF_VERSION = "2.18.1";
const TF_OFFICIAL_BASE = `https://storage.googleapis.com/tensorflow/versions/${TF_VERSION}`;

/**
 * GitHub Releases base URL for isidorus tensorflow binaries (Linux only).
 * Downloads a timestamped release tarball containing libtensorflow/linux-avx2 and libtensorflow/linux-legacy.
 * For custom releases, set LIBTF_RELEASE_TAG environment variable.
 */
const LIBTF_RELEASE_TAG =
  process.env.LIBTF_RELEASE_TAG || "tensorflow-binaries-latest";
const ISIDORUS_RELEASES = `https://github.com/A-KGeorge/isidorus/releases/download/${LIBTF_RELEASE_TAG}`;

const __dirname = dirname(fileURLToPath(import.meta.url));
const PACKAGE_DIR = dirname(__dirname);
const WORKSPACE_ROOT = dirname(dirname(PACKAGE_DIR)); // Go up from packages/cpu to workspace root
const LIBTF_DIR = join(PACKAGE_DIR, "libtf");

// ── CPU capability detection ──────────────────────────────────────────────────

/**
 * Read /proc/cpuinfo and return the set of CPU flags for the first processor.
 * Returns an empty Set on any error (e.g. non-Linux, no /proc).
 */
function readCpuFlags() {
  try {
    const cpuinfo = readFileSync("/proc/cpuinfo", "utf8");
    const match = cpuinfo.match(/^flags\s*:\s*(.+)$/m);
    if (!match) return new Set();
    return new Set(match[1].trim().split(/\s+/));
  } catch {
    return new Set();
  }
}

/**
 * Choose the best available MKL variant for this CPU.
 * Returns: "mkl-avx2" | "mkl" | "cpu"
 *
 *   mkl-avx2 — AVX2 + FMA required (Intel Haswell 2013+, AMD Zen 1 2017+)
 *   mkl      — any x86_64; MKL without AVX2 specialisation
 *   cpu      — fall back to official libtensorflow (no MKL)
 */
function detectLinuxVariant() {
  const forced = process.env.LIBTF_VARIANT;
  if (forced) {
    if (!["mkl-avx2", "mkl", "cpu"].includes(forced)) {
      console.warn(
        `[isidorus] Unknown LIBTF_VARIANT="${forced}". ` +
          `Valid: mkl-avx2 | mkl | cpu. Falling back to auto-detect.`,
      );
    } else {
      console.log(`[isidorus] LIBTF_VARIANT=${forced} (forced)`);
      return forced;
    }
  }

  const flags = readCpuFlags();
  if (flags.has("avx2") && flags.has("fma")) return "mkl-avx2";
  if (flags.size > 0) return "mkl";
  // /proc/cpuinfo unreadable — use official build.
  return "cpu";
}

// ── Platform spec ─────────────────────────────────────────────────────────────

function getPlatformSpec() {
  const os = platform();
  const cpu = arch();

  // ── Linux ────────────────────────────────────────────────────────────────
  if (os === "linux") {
    const archStr = cpu === "arm64" ? "aarch64" : "x86_64";
    const officialTarball = `libtensorflow-cpu-linux-${archStr}.tar.gz`;
    const officialUrl = `${TF_OFFICIAL_BASE}/${officialTarball}`;

    let primaryUrl = officialUrl;
    let primaryLabel = "official";
    let variantTag = "cpu";

    // For x86_64, download from isidorus releases (contains both AVX2 and legacy variants).
    if (cpu === "x86_64") {
      variantTag = detectLinuxVariant();
      if (variantTag === "mkl-avx2") {
        primaryUrl = `${ISIDORUS_RELEASES}/tensorflow-binaries-avx2.tar.gz`;
        primaryLabel = "AVX2 + FMA optimized (isidorus release)";
      } else if (variantTag === "mkl") {
        primaryUrl = `${ISIDORUS_RELEASES}/tensorflow-binaries-legacy.tar.gz`;
        primaryLabel = "Legacy CPU optimized (isidorus release)";
      }
    }

    return {
      libFile: "libtensorflow.so",
      tarball: officialTarball,
      primaryUrl,
      primaryLabel,
      // Provide official URL as fallback only when a custom build was attempted.
      fallbackUrl: variantTag !== "cpu" ? officialUrl : null,
      extractCmd: (src, dst) => {
        // The isidorus tarballs contain linux-avx2/ or linux-legacy/ at the top level
        // We extract to a temp location, then move the contents up
        return variantTag !== "cpu"
          ? `mkdir -p '${dst}' && tar -C '${dst}' -xzf '${src}' && mv '${dst}'/{linux-avx2,linux-legacy}/* '${dst}'/ 2>/dev/null || true`
          : `tar -C '${dst}' -xzf '${src}'`;
      },
      postExtract: async () => {
        try {
          await execAsync("ldconfig " + join(LIBTF_DIR, "lib"));
        } catch {}
      },
      winPath: null,
      variantTag,
    };
  }

  // ── macOS ────────────────────────────────────────────────────────────────
  if (os === "darwin") {
    const archStr = cpu === "arm64" ? "arm64" : "x86_64";
    const tarball = `libtensorflow-cpu-darwin-${archStr}.tar.gz`;
    return {
      libFile: "libtensorflow.dylib",
      tarball,
      primaryUrl: `${TF_OFFICIAL_BASE}/${tarball}`,
      primaryLabel: "official",
      fallbackUrl: null,
      extractCmd: (src, dst) => `tar -C '${dst}' -xzf '${src}'`,
      postExtract: async () => {},
      winPath: null,
      variantTag: "cpu",
    };
  }

  // ── Windows ──────────────────────────────────────────────────────────────
  if (os === "win32") {
    const tarball = "libtensorflow-cpu-windows-x86_64.zip";
    return {
      libFile: "tensorflow.dll",
      tarball,
      primaryUrl: `${TF_OFFICIAL_BASE}/${tarball}`,
      primaryLabel: "official",
      fallbackUrl: null,
      extractCmd: (src, dst) =>
        `powershell -NoProfile -Command "Expand-Archive -Path '${src}' -DestinationPath '${dst}' -Force"`,
      postExtract: async () => {},
      winPath: join(LIBTF_DIR, "lib"),
      variantTag: "cpu",
    };
  }

  throw new Error(`[isidorus] Unsupported platform: ${os}`);
}

// ── Skip conditions ───────────────────────────────────────────────────────────

function getInstalledVariant() {
  try {
    const configPath = join(PACKAGE_DIR, ".libtf-config.json");
    if (!existsSync(configPath)) return null;
    const config = JSON.parse(readFileSync(configPath, "utf8"));
    return config.variant;
  } catch {
    return null;
  }
}

function shouldSkip(spec) {
  if (process.env.SKIP_LIBTF_DOWNLOAD === "1") {
    console.log("[isidorus] SKIP_LIBTF_DOWNLOAD=1 — skipping.");
    return true;
  }
  if (process.env.LIBTENSORFLOW_PATH) {
    console.log(
      `[isidorus] LIBTENSORFLOW_PATH is set — skipping (${process.env.LIBTENSORFLOW_PATH}).`,
    );
    return true;
  }

  const libPath = join(LIBTF_DIR, "lib", spec.libFile);
  if (!existsSync(libPath)) {
    return false; // Library doesn't exist, need to download
  }

  // Library exists — check if it's the correct variant for this CPU
  const installedVariant = getInstalledVariant();
  if (installedVariant === spec.variantTag) {
    console.log(
      `[isidorus] libtensorflow ${spec.variantTag} already installed at ${LIBTF_DIR}.`,
    );
    return true;
  }

  // Variant mismatch — re-download the correct one
  console.log(
    `[isidorus] Variant mismatch: installed ${installedVariant}, but CPU supports ${spec.variantTag}.`,
  );
  console.log(`[isidorus] Re-downloading correct variant...`);

  // Clean up old binaries before re-downloading
  try {
    const libDir = join(LIBTF_DIR, "lib");
    if (existsSync(libDir)) {
      for (const file of readdirSync(libDir)) {
        if (file.startsWith("libtensorflow")) {
          unlinkSync(join(libDir, file));
        }
      }
    }
  } catch (e) {
    console.warn(`[isidorus] Warning cleaning old binaries: ${e.message}`);
  }

  return false; // Force re-download
}

// ── Library relocation ────────────────────────────────────────────────────────

function moveLibrariesToPrebuilds(spec) {
  const os = platform();
  if (os === "win32") return;

  const libDir = join(LIBTF_DIR, "lib");
  if (!existsSync(libDir)) return;

  const libFilePatterns =
    os === "linux"
      ? [".so", ".so.2", ".so.2.18.1"]
      : [".dylib", ".dylib.2", ".dylib.2.18.1"];

  const cpu = arch();
  const platformStr = os === "darwin" ? `darwin-${cpu}` : `linux-${cpu}`;
  const prebuildsDir = join(PACKAGE_DIR, "prebuilds", platformStr);
  mkdirSync(prebuildsDir, { recursive: true });

  try {
    let movedCount = 0;
    for (const file of readdirSync(libDir)) {
      if (
        file.startsWith("libtensorflow") &&
        libFilePatterns.some((p) => file.includes(p))
      ) {
        const src = join(libDir, file);
        const dst = join(prebuildsDir, file);
        const symlink = join(libDir, file);
        try {
          renameSync(src, dst);
          try {
            if (existsSync(symlink)) unlinkSync(symlink);
            symlinkSync(
              join("..", "..", "prebuilds", platformStr, file),
              symlink,
            );
          } catch (e) {
            console.warn(
              `[isidorus] symlink warning for ${file}: ${e.message}`,
            );
          }
          movedCount++;
        } catch (e) {
          console.warn(`[isidorus] failed to move ${file}: ${e.message}`);
        }
      }
    }
    if (movedCount > 0)
      console.log(
        `  Moved ${movedCount} library files to prebuilds/${platformStr}/`,
      );
  } catch (e) {
    console.warn(`[isidorus] library move warning: ${e.message}`);
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
        let downloaded = 0,
          lastPct = -1;

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
      "[isidorus] Set LIBTENSORFLOW_PATH to your TF directory and rebuild.",
    );
    return;
  }

  if (shouldSkip(spec)) return;

  // User URL override takes precedence over everything.
  const overrideUrl = process.env.LIBTF_DOWNLOAD_URL ?? null;
  const downloadUrl = overrideUrl ?? spec.primaryUrl;
  const tmpFile = join(tmpdir(), spec.tarball);

  console.log(`\n[isidorus] Installing libtensorflow ${TF_VERSION}...`);
  console.log(`  Platform : ${platform()}-${arch()}`);
  if (spec.variantTag !== "cpu" && !overrideUrl) {
    console.log(`  Variant  : ${spec.variantTag} (${spec.primaryLabel})`);
    console.log(`  Override : LIBTF_VARIANT=cpu to force the official build`);
  }
  console.log(`  Dest     : ${LIBTF_DIR}\n`);

  mkdirSync(LIBTF_DIR, { recursive: true });

  // ── Attempt primary download ──────────────────────────────────────────────
  let downloadedUrl = downloadUrl;
  try {
    await downloadFile(downloadUrl, tmpFile);
  } catch (primaryErr) {
    if (spec.fallbackUrl && !overrideUrl) {
      console.warn(
        `\n[isidorus] Custom build unavailable: ${primaryErr.message}`,
      );
      console.warn(
        `[isidorus] Falling back to official libtensorflow (no MKL)...`,
      );
      console.warn(
        `[isidorus] Tip: set LIBTF_VARIANT=cpu to skip the custom build.\n`,
      );
      try {
        await downloadFile(spec.fallbackUrl, tmpFile);
        downloadedUrl = spec.fallbackUrl;
      } catch (fallbackErr) {
        console.error(
          `\n[isidorus] Fallback also failed: ${fallbackErr.message}`,
        );
        console.error(`[isidorus] Install manually:\n  ${spec.fallbackUrl}`);
        return;
      }
    } else {
      console.error(`\n[isidorus] Download failed: ${primaryErr.message}`);
      console.error(`[isidorus] Install manually:\n  ${downloadUrl}`);
      return;
    }
  }

  // ── Extract ───────────────────────────────────────────────────────────────
  console.log(`  Extracting to ${LIBTF_DIR}...`);
  try {
    await execAsync(spec.extractCmd(tmpFile, LIBTF_DIR));
  } catch (e) {
    console.error(`[isidorus] Extraction failed: ${e.message}`);
    console.error(
      `[isidorus] Archive saved at ${tmpFile} — extract manually to ${LIBTF_DIR}.`,
    );
    return;
  }

  await spec.postExtract();
  moveLibrariesToPrebuilds(spec);

  // ── Write config ──────────────────────────────────────────────────────────
  writeFileSync(
    join(PACKAGE_DIR, ".libtf-config.json"),
    JSON.stringify(
      {
        version: TF_VERSION,
        variant: spec.variantTag,
        installedAt: new Date().toISOString(),
        sourceUrl: downloadedUrl,
        libtfDir: LIBTF_DIR,
        platform: platform(),
        arch: arch(),
      },
      null,
      2,
    ),
  );

  // ── Verify ────────────────────────────────────────────────────────────────
  const libPath = join(LIBTF_DIR, "lib", spec.libFile);
  if (!existsSync(libPath)) {
    console.error(`[isidorus] ⚠  ${spec.libFile} not found after extraction.`);
    console.error(`[isidorus]    Expected: ${libPath}`);
    console.error(
      `[isidorus]    Archive structure may differ — check ${LIBTF_DIR}.`,
    );
    return;
  }

  console.log(
    `\n[isidorus] libtensorflow ${TF_VERSION} installed successfully.`,
  );
  console.log(`  Library  : ${libPath}`);
  if (spec.variantTag !== "cpu") {
    console.log(`  Variant  : ${spec.variantTag} — MKL-optimized`);
  }
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
  // Never throw from postinstall — a failed optional download must not
  // break `npm install` for the parent project.
  console.warn(`[isidorus] post-install warning: ${e.message}`);
});
