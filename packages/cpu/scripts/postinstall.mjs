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
  lstatSync,
  statSync,
  chmodSync,
  createReadStream,
  rmSync,
} from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { platform, arch, tmpdir } from "node:os";
import { get } from "node:https";
import crypto from "node:crypto";
import { mkdtemp } from "node:fs/promises";
import { exec, execSync } from "node:child_process";
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
 * Validates flag format (alphanumeric + underscore only).
 */
function readCpuFlags() {
  try {
    const cpuinfo = readFileSync("/proc/cpuinfo", "utf8");
    const match = cpuinfo.match(/^flags\s*:\s*(.+)$/m);
    if (!match) return new Set();

    // Validate that flags match expected format (alphanumeric + underscore)
    const flags = match[1]
      .trim()
      .split(/\s+/)
      .filter((f) => /^[a-z0-9_]+$/.test(f));

    return new Set(flags);
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
const VALID_VARIANTS = { "mkl-avx2": true, mkl: true, cpu: true };

function detectLinuxVariant() {
  const forced = process.env.LIBTF_VARIANT;
  if (forced) {
    // Strict enum-based validation
    if (!VALID_VARIANTS[forced]) {
      console.warn(
        `[isidorus] Unknown LIBTF_VARIANT="${forced}". ` +
          `Valid: mkl-avx2 | mkl | cpu. Falling back to auto-detect.`,
      );
      return detectVariantFromCpu();
    }
    console.log(`[isidorus] LIBTF_VARIANT=${forced} (forced)`);
    return forced;
  }

  return detectVariantFromCpu();
}

function detectVariantFromCpu() {
  const flags = readCpuFlags();
  console.log(
    `[isidorus] CPU flags detected: ${
      flags.size > 0 ? Array.from(flags).join(" ") : "(none)"
    }`,
  );

  if (flags.has("avx2") && flags.has("fma")) {
    console.log(`[isidorus] AVX2 + FMA detected → using mkl-avx2`);
    return "mkl-avx2";
  }
  if (flags.size > 0) {
    console.log(`[isidorus] Generic x86_64 detected → using mkl (legacy)`);
    return "mkl";
  }

  // /proc/cpuinfo unreadable — use official build.
  console.log(`[isidorus] Could not detect CPU flags → using official build`);
  return "cpu";
}

// ── Security: URL validation ──────────────────────────────────────────────────

const ALLOWED_DOWNLOAD_HOSTS = new Set([
  "github.com",
  "storage.googleapis.com",
  "releases.github.com",
  "codeload.github.com",
]);

const ALLOWED_URL_PATTERNS = [
  /^https:\/\/(github\.com|releases\.github\.com|codeload\.github\.com)\/A-KGeorge\/isidorus/,
  /^https:\/\/storage\.googleapis\.com\/tensorflow\//,
];

function validateDownloadUrl(urlString) {
  try {
    const url = new URL(urlString);

    // 1. Must be HTTPS only
    if (url.protocol !== "https:") {
      throw new Error(
        "Only HTTPS URLs are allowed (got: " + url.protocol + ")",
      );
    }

    // 2. Host must be whitelisted
    if (!ALLOWED_DOWNLOAD_HOSTS.has(url.hostname)) {
      throw new Error(
        `Download host not whitelisted: ${url.hostname}. ` +
          `Allowed: ${Array.from(ALLOWED_DOWNLOAD_HOSTS).join(", ")}`,
      );
    }

    // 3. Path must match expected patterns
    let patternMatched = false;
    for (const pattern of ALLOWED_URL_PATTERNS) {
      if (pattern.test(urlString)) {
        patternMatched = true;
        break;
      }
    }
    if (!patternMatched) {
      throw new Error(
        `URL path does not match expected patterns: ${url.pathname}`,
      );
    }

    // 4. URL must be for expected file types
    const filename = url.pathname.split("/").pop() || "";
    if (
      !/^(libtensorflow|tensorflow-binaries)[^/]*\.(tar\.gz|zip)$/.test(
        filename,
      )
    ) {
      throw new Error(`Unexpected filename in URL: ${filename}`);
    }

    return true;
  } catch (e) {
    if (e instanceof TypeError) {
      throw new Error(`Invalid URL format: ${urlString}`);
    }
    throw e;
  }
}

// ── Security: Checksum verification ───────────────────────────────────────────

/**
 * SHA256 checksums for all supported library artifacts.
 * IMPORTANT: These must be updated with actual checksums from official sources.
 * Use: sha256sum <file> or shasum -a 256 <file>
 */
const CHECKSUM_MAP = {
  // Linux variants
  "libtensorflow-cpu-linux-x86_64.tar.gz":
    "b692795f3ad198c531b02aeb2bc8146568d24aaf6a5dbf5faa43907c4028fd73",
  "tensorflow-binaries-avx2.tar.gz":
    "e642d477d7de5fd90ffa8ffee183c0e36aa8d3705bc372b14af08f841dbf15fa",
  "tensorflow-binaries-legacy.tar.gz":
    "50ba70a5d4163c08bdce31bf1f078264548502d03c87fedaded878a3add9f68f",
  // macOS variants
  "libtensorflow-cpu-darwin-arm64.tar.gz":
    "61258fbcc8ff57d2868fa56f20edc06443a29eb2169b9f04515a405d5f1432ec",
  // Windows variants
  "libtensorflow-cpu-windows-x86_64.zip":
    "28acdcea6c6b34828cf0e95e67802b0f3577d51bc2e8915de811b7aa0b04452d",
};

async function verifyChecksum(filePath, expectedHash) {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash("sha256");
    const stream = createReadStream(filePath);

    stream.on("data", (data) => hash.update(data));
    stream.on("end", () => {
      const computed = hash.digest("hex");
      if (computed.toLowerCase() !== expectedHash.toLowerCase()) {
        reject(
          new Error(
            `[isidorus] CHECKSUM MISMATCH!\n` +
              `File: ${filePath}\n` +
              `Expected: ${expectedHash}\n` +
              `Got:      ${computed}\n` +
              `This could indicate a corrupted download or a MITM attack.`,
          ),
        );
      } else {
        resolve(computed);
      }
    });

    stream.on("error", reject);
  });
}

// ── Platform spec ─────────────────────────────────────────────────────────────

function getPlatformSpec() {
  const os = platform();
  const cpu = arch();

  // ── Linux ────────────────────────────────────────────────────────────────
  if (os === "linux") {
    const archStr = cpu === "arm64" ? "aarch64" : "x86_64";

    // aarch64 builds not available
    if (archStr === "aarch64") {
      throw new Error(
        `[isidorus] TensorFlow C library is not available for Linux aarch64.\n` +
          `Please provide a TensorFlow build manually or use SKIP_LIBTF_DOWNLOAD=1`,
      );
    }

    const officialTarball = `libtensorflow-cpu-linux-${archStr}.tar.gz`;
    const officialUrl = `${TF_OFFICIAL_BASE}/${officialTarball}`;

    let primaryUrl = officialUrl;
    let primaryLabel = "official";
    let variantTag = "cpu";
    let fallbackUrl = null;

    // For x64 (x86_64), ALWAYS try isidorus releases first (with fallback to official)
    if (cpu === "x64") {
      variantTag = detectLinuxVariant();

      // Determine which GitHub release to try first
      let githubVariant = variantTag;
      if (variantTag === "cpu") {
        // CPU detection failed, but we'll still try the legacy variant from GitHub
        // It's more likely to work than the official build
        githubVariant = "mkl";
      }

      if (githubVariant === "mkl-avx2") {
        primaryUrl = `${ISIDORUS_RELEASES}/tensorflow-binaries-avx2.tar.gz`;
        primaryLabel = "AVX2 + FMA optimized (isidorus release)";
      } else if (githubVariant === "mkl") {
        primaryUrl = `${ISIDORUS_RELEASES}/tensorflow-binaries-legacy.tar.gz`;
        primaryLabel = "Legacy CPU optimized (isidorus release)";
      }

      // Always provide official as fallback for x86_64
      fallbackUrl = officialUrl;
    }

    return {
      libFile: "libtensorflow.so",
      tarball: officialTarball,
      primaryUrl,
      primaryLabel,
      fallbackUrl,
      extractCmd: (src, dst) => {
        // The isidorus tarballs contain linux-avx2/ or linux-legacy/ at the top level
        // Determine which variant directory to extract based on the URL
        if (primaryUrl.includes("avx2")) {
          return `mkdir -p '${dst}' && tar -C '${dst}' -xzf '${src}' && [ -d '${dst}/linux-avx2' ] && mv '${dst}/linux-avx2'/* '${dst}/' || true`;
        } else if (primaryUrl.includes("legacy")) {
          return `mkdir -p '${dst}' && tar -C '${dst}' -xzf '${src}' && [ -d '${dst}/linux-legacy' ] && mv '${dst}/linux-legacy'/* '${dst}/' || true`;
        }
        return `tar -C '${dst}' -xzf '${src}'`;
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

    // x86_64 builds not available
    if (archStr === "x86_64") {
      throw new Error(
        `[isidorus] TensorFlow C library is not available for macOS x86_64.\n` +
          `Please provide a TensorFlow build manually or use SKIP_LIBTF_DOWNLOAD=1`,
      );
    }

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
            // Security: Safely validate and remove existing symlink
            try {
              const stats = lstatSync(symlink);
              if (stats.isSymbolicLink()) {
                unlinkSync(symlink);
              } else if (stats.isFile()) {
                throw new Error(
                  `Cannot create symlink: regular file exists at ${symlink}`,
                );
              }
            } catch (e) {
              if (e.code !== "ENOENT") throw e; // File doesn't exist, OK
            }

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

const MAX_REDIRECTS = 5;
const ALLOWED_REDIRECT_HOSTS = new Set([
  "github.com",
  "storage.googleapis.com",
  "releases.github.com",
  "codeload.github.com",
]);

function downloadFile(url, dest) {
  return new Promise((resolve, reject) => {
    console.log(`  Downloading: ${url}`);
    const file = createWriteStream(dest);

    const request = (u, redirectCount = 0) =>
      get(u, (res) => {
        // Security: Validate redirects
        if (res.statusCode === 301 || res.statusCode === 302) {
          if (redirectCount >= MAX_REDIRECTS) {
            reject(new Error(`Too many redirects (max: ${MAX_REDIRECTS})`));
            return;
          }

          const redirectUrl = res.headers.location;
          if (!redirectUrl) {
            reject(new Error("Redirect without Location header"));
            return;
          }

          // Validate redirect destination
          try {
            const parsed = new URL(redirectUrl, u);
            if (!ALLOWED_REDIRECT_HOSTS.has(parsed.hostname)) {
              reject(
                new Error(`Redirect to untrusted host: ${parsed.hostname}`),
              );
              return;
            }
            console.log(
              `  Following redirect (${redirectCount + 1}/${MAX_REDIRECTS})...`,
            );
            request(redirectUrl, redirectCount + 1);
          } catch (e) {
            reject(new Error(`Invalid redirect URL: ${e.message}`));
          }
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

// ── File validation ──────────────────────────────────────────────────────────

function validateLibraryFile(libPath) {
  try {
    const stats = statSync(libPath);

    // Check file size is reasonable (not corrupted/partial)
    const MIN_SIZE = 1024 * 1024; // 1MB minimum
    if (stats.size < MIN_SIZE) {
      throw new Error(`Library file suspiciously small: ${stats.size} bytes`);
    }

    // Check file permissions are readable
    if ((stats.mode & 0o400) === 0) {
      // S_IRUSR = 0o400
      throw new Error("Library file is not readable");
    }

    // For Linux, verify it's an ELF binary
    if (platform() === "linux") {
      try {
        const fileOutput = execSync(`file "${libPath.replace(/"/g, '\\"')}"`, {
          encoding: "utf8",
          stdio: ["pipe", "pipe", "ignore"],
        });

        if (
          !fileOutput.includes("ELF") &&
          !fileOutput.includes("shared object")
        ) {
          throw new Error("File is not an ELF shared object");
        }
      } catch (e) {
        // 'file' command may not be available; skip this check
        if (!/command not found|not found/.test(e.message)) {
          throw e;
        }
      }
    }

    return true;
  } catch (e) {
    throw new Error(`Library validation failed: ${e.message}`);
  }
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

  // Security: Validate environment variable overrides
  const overrideUrl = process.env.LIBTF_DOWNLOAD_URL ?? null;
  if (overrideUrl) {
    try {
      validateDownloadUrl(overrideUrl);
    } catch (e) {
      console.error(`[isidorus] ${e.message}`);
      if (spec.fallbackUrl) {
        console.error(
          "[isidorus] Invalid LIBTF_DOWNLOAD_URL override. Using fallback URL instead.",
        );
      } else {
        console.error("[isidorus] No fallback URL available. Aborting.");
        return;
      }
    }
  }

  const downloadUrl = overrideUrl ?? spec.primaryUrl;

  // Validate primary URL
  try {
    validateDownloadUrl(downloadUrl);
  } catch (e) {
    console.error(`[isidorus] Invalid download URL: ${e.message}`);
    return;
  }

  // Security: Create secure temporary directory
  let tmpDir;
  let tmpFile;
  try {
    tmpDir = await mkdtemp(join(tmpdir(), "isidorus-tf-"));
    tmpFile = join(tmpDir, spec.tarball);
  } catch (e) {
    console.error(
      `[isidorus] Failed to create secure temp directory: ${e.message}`,
    );
    return;
  }

  console.log(`\n[isidorus] Installing libtensorflow ${TF_VERSION}...`);
  console.log(`  Platform  : ${platform()}-${arch()}`);
  console.log(`  Variant   : ${spec.variantTag}`);
  console.log(`  Label     : ${spec.primaryLabel}`);
  console.log(`  Download  : ${downloadUrl.split("?")[0]}`); // Hide query params if any
  console.log(`  Dest      : ${LIBTF_DIR}\n`);

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
        // Validate fallback URL before download
        validateDownloadUrl(spec.fallbackUrl);
        await downloadFile(spec.fallbackUrl, tmpFile);
        downloadedUrl = spec.fallbackUrl;
      } catch (fallbackErr) {
        console.error(
          `\n[isidorus] Fallback also failed: ${fallbackErr.message}`,
        );
        console.error(`[isidorus] Install manually:\n  ${spec.fallbackUrl}`);
        try {
          rmSync(tmpDir, { recursive: true, force: true });
        } catch {}
        return;
      }
    } else {
      console.error(`\n[isidorus] Download failed: ${primaryErr.message}`);
      console.error(`[isidorus] Install manually:\n  ${downloadUrl}`);
      try {
        rmSync(tmpDir, { recursive: true, force: true });
      } catch {}
      return;
    }
  }

  // ── Verify Checksum ────────────────────────────────────────────────────────
  console.log(`  Verifying checksum...`);
  try {
    const expectedHash = CHECKSUM_MAP[spec.tarball];
    if (!expectedHash || expectedHash.startsWith("TODO_")) {
      if (process.env.SKIP_CHECKSUM_VERIFY === "1") {
        console.warn(
          `[isidorus] WARNING: Checksum not yet available, skipping verification!`,
        );
      } else {
        throw new Error(
          `No checksum defined for ${spec.tarball}. ` +
            `Set SKIP_CHECKSUM_VERIFY=1 to proceed at your own risk.`,
        );
      }
    } else if (process.env.SKIP_CHECKSUM_VERIFY === "1") {
      console.warn(`[isidorus] WARNING: Skipping checksum verification!`);
    } else {
      await verifyChecksum(tmpFile, expectedHash);
      console.log(`  ✓ Checksum verified`);
    }
  } catch (e) {
    console.error(`[isidorus] ${e.message}`);
    try {
      rmSync(tmpDir, { recursive: true, force: true });
    } catch {}
    return;
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

  // ── Clean up extracted files to conserve space (Linux only) ──────────────
  // Windows keeps binaries in lib/ for PATH injection
  // macOS uses lib/ for RPATH resolution
  // Only Linux can safely remove lib/ since binaries are in prebuilds/
  if (platform() === "linux") {
    try {
      const libDir = join(LIBTF_DIR, "lib");
      const includeDir = join(LIBTF_DIR, "include");

      // Aggressively remove lib and include directories
      if (existsSync(libDir)) {
        await execAsync(`rm -rf '${libDir}'`);
        console.log(`  Removed lib directory to conserve space`);
      }

      if (existsSync(includeDir)) {
        await execAsync(`rm -rf '${includeDir}'`);
        console.log(`  Removed include directory to conserve space`);
      }
    } catch (e) {
      console.warn(
        `[isidorus] Warning cleaning up extracted files: ${e.message}`,
      );
    }
  }

  // ── Cleanup secure temp directory ──────────────────────────────────────────
  try {
    if (tmpDir && existsSync(tmpDir)) {
      rmSync(tmpDir, { recursive: true, force: true });
    }
  } catch (e) {
    console.warn(`[isidorus] Warning cleaning temp dir: ${e.message}`);
  }

  // ── Determine actual variant that was downloaded ──────────────────────────
  let actualVariant = spec.variantTag;
  if (downloadedUrl && downloadedUrl.includes("isidorus")) {
    if (downloadedUrl.includes("avx2")) {
      actualVariant = "mkl-avx2";
    } else if (downloadedUrl.includes("legacy")) {
      actualVariant = "mkl";
    }
  }

  // ── Write config ──────────────────────────────────────────────────────────
  // Security: Protect config file with restricted permissions
  const configPath = join(PACKAGE_DIR, ".libtf-config.json");
  const configData = JSON.stringify(
    {
      version: TF_VERSION,
      variant: actualVariant,
      installedAt: new Date().toISOString(),
      sourceUrl: downloadedUrl,
      libtfDir: LIBTF_DIR,
      platform: platform(),
      arch: arch(),
    },
    null,
    2,
  );

  writeFileSync(configPath, configData, { mode: 0o600 });
  chmodSync(configPath, 0o600);

  // ── Validate library file ──────────────────────────────────────────────────
  const libPath = join(LIBTF_DIR, "lib", spec.libFile);
  if (!existsSync(libPath)) {
    console.error(`[isidorus] ⚠  ${spec.libFile} not found after extraction.`);
    console.error(`[isidorus]    Expected: ${libPath}`);
    console.error(
      `[isidorus]    Archive structure may differ — check ${LIBTF_DIR}.`,
    );
    return;
  }

  // Security: Validate extracted library file integrity
  try {
    validateLibraryFile(libPath);
    console.log(`  ✓ Library file validation passed`);
  } catch (e) {
    console.error(`[isidorus] ${e.message}`);
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
  // Improved error handling: provide helpful context and solutions
  console.error(`\n[isidorus] ========================================`);
  console.error(`[isidorus] CRITICAL: post-install failed!`);
  console.error(`[isidorus] ========================================`);
  console.error(`[isidorus] Error: ${e.message}\n`);
  console.error(
    `[isidorus] This package requires TensorFlow C library to function.`,
  );
  console.error(`[isidorus]`);
  console.error(`[isidorus] Troubleshooting steps:`);
  console.error(
    `[isidorus]   1. Retry: rm -rf node_modules/@isidorus && npm install`,
  );
  console.error(
    `[isidorus]   2. Manual: LIBTENSORFLOW_PATH=/path/to/tf npm install`,
  );
  console.error(
    `[isidorus]   3. Skip (advanced): SKIP_LIBTF_DOWNLOAD=1 npm install`,
  );
  console.error(
    `[isidorus]      (you must provide TensorFlow via LIBTENSORFLOW_PATH)`,
  );
  console.error(
    `[isidorus]   4. Report: https://github.com/A-KGeorge/isidorus/issues`,
  );
  console.error(`[isidorus] ========================================\n`);

  // Allow bypassing for specific scenarios (CI environments, etc.)
  if (process.env.ISIDORUS_ALLOW_MISSING_TF === "1") {
    console.warn(
      `[isidorus] Proceeding despite error (ISIDORUS_ALLOW_MISSING_TF=1)`,
    );
  } else {
    // Exit with error code - this is a critical failure
    process.exit(1);
  }
});
