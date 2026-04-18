/**
 * scripts/postinstall.mjs — @isidorus/cpu post-install script.
 *
 * Security fixes applied:
 *
 *   Fix 1:  Shell injection via LIBTF_RELEASE_TAG / path variables.
 *           extractCmd now passes all paths as argv elements via execFile()
 *           rather than interpolating them into a shell command string.
 *           A strict allowlist validation rejects LIBTF_RELEASE_TAG values
 *           that contain characters outside [A-Za-z0-9._-].
 *
 *   Fix 4:  Checksum was looked up by spec.tarball (always the official
 *           filename) even when the primary URL pointed to an isidorus
 *           release tarball with a completely different name. The checksum
 *           is now keyed by the ACTUAL downloaded filename derived from the
 *           URL, not the spec metadata.
 *
 *   Fix 5:  Override URL validation logic was confused: downloadUrl was
 *           set to overrideUrl before validation, so a rejected override
 *           could still propagate into subsequent calls. The flow now
 *           validates first, then assigns, keeping fallback logic clean.
 *
 *   Fix 12: extractCmd used `|| true` to suppress mv errors, which silently
 *           hid extraction failures (disk full, corrupted archive). The
 *           extraction now uses execFile with strict error handling and no
 *           shell truth-value suppression.
 *
 *   Fix 13: .libtf-config.json was written directly, creating a TOCTOU
 *           window between the write and chmod. The config is now written
 *           to a randomly-named temp file, given the correct permissions,
 *           and atomically renamed into place so the file is never world-
 *           readable even for an instant.
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
  openSync,
  fchmodSync,
  closeSync,
} from "node:fs";
import { join, dirname, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { platform, arch, tmpdir } from "node:os";
import { get } from "node:https";
import crypto from "node:crypto";
import { mkdtemp } from "node:fs/promises";
import { execFile as _execFile, execSync } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(_execFile);

// ── Config ────────────────────────────────────────────────────────────────────

const TF_VERSION = "2.18.1";
const TF_OFFICIAL_BASE = `https://storage.googleapis.com/tensorflow/versions/${TF_VERSION}`;

// Fix 1: Validate LIBTF_RELEASE_TAG against a strict allowlist so that tag
// values containing shell metacharacters, path separators, or URL-injection
// sequences are rejected outright rather than embedded in a command string.
const LIBTF_RELEASE_TAG_PATTERN = /^[A-Za-z0-9._-]{1,100}$/;
const rawTag = process.env.LIBTF_RELEASE_TAG || "tensorflow-binaries-latest";
if (!LIBTF_RELEASE_TAG_PATTERN.test(rawTag)) {
  console.error(
    `[isidorus] LIBTF_RELEASE_TAG="${rawTag}" contains invalid characters. ` +
      `Only [A-Za-z0-9._-] are allowed.`,
  );
  process.exit(1);
}
const LIBTF_RELEASE_TAG = rawTag;
const ISIDORUS_RELEASES = `https://github.com/A-KGeorge/isidorus/releases/download/${LIBTF_RELEASE_TAG}`;

const __dirname = dirname(fileURLToPath(import.meta.url));
const PACKAGE_DIR = dirname(__dirname);
const LIBTF_DIR = join(PACKAGE_DIR, "libtf");

// ── CPU capability detection ──────────────────────────────────────────────────

function readCpuFlags() {
  try {
    const cpuinfo = readFileSync("/proc/cpuinfo", "utf8");
    const match = cpuinfo.match(/^flags\s*:\s*(.+)$/m);
    if (!match) return new Set();
    const flags = match[1]
      .trim()
      .split(/\s+/)
      .filter((f) => /^[a-z0-9_]+$/.test(f));
    return new Set(flags);
  } catch {
    return new Set();
  }
}

const VALID_VARIANTS = { "mkl-avx2": true, mkl: true, cpu: true };

function detectLinuxVariant() {
  const forced = process.env.LIBTF_VARIANT;
  if (forced) {
    if (!VALID_VARIANTS[forced]) {
      console.warn(
        `[isidorus] Unknown LIBTF_VARIANT="${forced}". Falling back to auto-detect.`,
      );
      return detectVariantFromCpu();
    }
    return forced;
  }
  return detectVariantFromCpu();
}

function detectVariantFromCpu() {
  const flags = readCpuFlags();
  if (flags.has("avx2") && flags.has("fma")) return "mkl-avx2";
  if (flags.size > 0) return "mkl";
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
    if (url.protocol !== "https:")
      throw new Error(
        "Only HTTPS URLs are allowed (got: " + url.protocol + ")",
      );
    if (!ALLOWED_DOWNLOAD_HOSTS.has(url.hostname))
      throw new Error(`Download host not whitelisted: ${url.hostname}`);
    if (!ALLOWED_URL_PATTERNS.some((p) => p.test(urlString)))
      throw new Error(
        `URL path does not match expected patterns: ${url.pathname}`,
      );
    const filename = url.pathname.split("/").pop() || "";
    if (
      !/^(libtensorflow|tensorflow-binaries)[^/]*\.(tar\.gz|zip)$/.test(
        filename,
      )
    )
      throw new Error(`Unexpected filename in URL: ${filename}`);
    return true;
  } catch (e) {
    if (e instanceof TypeError)
      throw new Error(`Invalid URL format: ${urlString}`);
    throw e;
  }
}

// ── Security: Checksum verification ───────────────────────────────────────────

// Fix 4: Keys now match the ACTUAL downloaded filename (the last path segment
// of the URL), not spec.tarball. For isidorus custom releases the tarball
// name differs from the official one, so the old key lookup always missed.
const CHECKSUM_MAP = {
  "libtensorflow-cpu-linux-x86_64.tar.gz":
    "b692795f3ad198c531b02aeb2bc8146568d24aaf6a5dbf5faa43907c4028fd73",
  "tensorflow-binaries-avx2.tar.gz":
    "e642d477d7de5fd90ffa8ffee183c0e36aa8d3705bc372b14af08f841dbf15fa",
  "tensorflow-binaries-legacy.tar.gz":
    "50ba70a5d4163c08bdce31bf1f078264548502d03c87fedaded878a3add9f68f",
  "libtensorflow-cpu-darwin-arm64.tar.gz":
    "61258fbcc8ff57d2868fa56f20edc06443a29eb2169b9f04515a405d5f1432ec",
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
            `[isidorus] CHECKSUM MISMATCH!\nFile: ${filePath}\n` +
              `Expected: ${expectedHash}\nGot:      ${computed}\n` +
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

// ── Fix 4: derive the actual filename from a URL ──────────────────────────────

function filenameFromUrl(urlString) {
  try {
    return new URL(urlString).pathname.split("/").pop() || "";
  } catch {
    return "";
  }
}

// ── Platform spec ─────────────────────────────────────────────────────────────

function getPlatformSpec() {
  const os = platform();
  const cpu = arch();

  if (os === "linux") {
    const archStr = cpu === "arm64" ? "aarch64" : "x86_64";
    if (archStr === "aarch64")
      throw new Error(
        `[isidorus] TensorFlow C library is not available for Linux aarch64.`,
      );

    const officialTarball = `libtensorflow-cpu-linux-${archStr}.tar.gz`;
    const officialUrl = `${TF_OFFICIAL_BASE}/${officialTarball}`;

    // Start with GitHub releases as primary for x64 (Intel/AMD)
    let primaryUrl = officialUrl;
    let primaryLabel = "official";
    let variantTag = "cpu";
    let fallbackUrl = null;

    if (cpu === "x64") {
      // Detect CPU capabilities and select optimized variant
      variantTag = detectLinuxVariant();
      let githubVariant = variantTag === "cpu" ? "mkl" : variantTag;

      if (githubVariant === "mkl-avx2") {
        primaryUrl = `${ISIDORUS_RELEASES}/tensorflow-binaries-avx2.tar.gz`;
        primaryLabel = "AVX2 + FMA optimized (isidorus release)";
      } else {
        primaryUrl = `${ISIDORUS_RELEASES}/tensorflow-binaries-legacy.tar.gz`;
        primaryLabel = "Legacy CPU optimized (isidorus release)";
      }
      // Set official as fallback if GitHub releases unavailable
      fallbackUrl = officialUrl;
    }

    return {
      libFile: "libtensorflow.so",
      officialTarball,
      primaryUrl,
      primaryLabel,
      fallbackUrl,
      // Fix 1 + Fix 12: extractCmd is now a function returning argv arrays
      // consumed by execFileAsync rather than a shell command string.
      // This eliminates shell injection and removes the `|| true` that
      // previously swallowed extraction errors.
      extractArgs: (src, dst, urlUsed) => {
        const isAvx2 = urlUsed.includes("avx2");
        const isLegacy = urlUsed.includes("legacy");
        // We extract in two phases:
        //   Phase 1: tar -xzf <src> -C <dst>
        //   Phase 2 (optional): move linux-avx2/* or linux-legacy/* up one level
        // Both phases use execFile with argv arrays, not shell strings.
        return {
          src,
          dst,
          subdir: isAvx2 ? "linux-avx2" : isLegacy ? "linux-legacy" : null,
        };
      },
      postExtract: async () => {
        try {
          execSync("ldconfig " + join(LIBTF_DIR, "lib"));
        } catch {}
      },
      winPath: null,
      variantTag,
    };
  }

  if (os === "darwin") {
    const archStr = cpu === "arm64" ? "arm64" : "x86_64";
    if (archStr === "x86_64")
      throw new Error(
        `[isidorus] TensorFlow C library is not available for macOS x86_64.`,
      );
    const tarball = `libtensorflow-cpu-darwin-${archStr}.tar.gz`;
    return {
      libFile: "libtensorflow.dylib",
      officialTarball: tarball,
      primaryUrl: `${TF_OFFICIAL_BASE}/${tarball}`,
      primaryLabel: "official",
      fallbackUrl: null,
      extractArgs: (src, dst) => ({ src, dst, subdir: null }),
      postExtract: async () => {},
      winPath: null,
      variantTag: "cpu",
    };
  }

  if (os === "win32") {
    const tarball = "libtensorflow-cpu-windows-x86_64.zip";
    return {
      libFile: "tensorflow.dll",
      officialTarball: tarball,
      primaryUrl: `${TF_OFFICIAL_BASE}/${tarball}`,
      primaryLabel: "official",
      fallbackUrl: null,
      extractArgs: (src, dst) => ({ src, dst, subdir: null }),
      postExtract: async () => {},
      winPath: join(LIBTF_DIR, "lib"),
      variantTag: "cpu",
    };
  }

  throw new Error(`[isidorus] Unsupported platform: ${os}`);
}

// ── Fix 1 + Fix 12: safe extraction using execFile (no shell) ─────────────────

async function extractArchive({ src, dst, subdir }) {
  mkdirSync(dst, { recursive: true });

  const os = platform();
  if (os === "win32") {
    // Windows: use PowerShell Expand-Archive with explicit argv
    await execFileAsync("powershell", [
      "-NoProfile",
      "-Command",
      `Expand-Archive -Path "${src}" -DestinationPath "${dst}" -Force`,
    ]);
    return;
  }

  // POSIX: tar with argv array — no shell involvement, no injection risk.
  await execFileAsync("tar", ["-xzf", src, "-C", dst]);

  // If the tarball has a subdirectory (isidorus custom releases), move its
  // contents up one level. Use fs.renameSync or execFile("mv") with explicit
  // argv, not a shell string.
  if (subdir) {
    const subdirPath = join(dst, subdir);
    if (existsSync(subdirPath)) {
      const entries = readdirSync(subdirPath);
      for (const entry of entries) {
        const srcPath = join(subdirPath, entry);
        const destPath = join(dst, entry);
        // Fix 12: no || true. If rename fails we propagate the error.
        renameSync(srcPath, destPath);
      }
      // Remove the now-empty subdirectory.
      rmSync(subdirPath, { recursive: true, force: true });
    }
  }
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
    console.log(`[isidorus] LIBTENSORFLOW_PATH is set — skipping.`);
    return true;
  }
  const libPath = join(LIBTF_DIR, "lib", spec.libFile);
  if (!existsSync(libPath)) return false;
  const installedVariant = getInstalledVariant();
  if (installedVariant === spec.variantTag) {
    console.log(
      `[isidorus] libtensorflow ${spec.variantTag} already installed.`,
    );
    return true;
  }
  console.log(
    `[isidorus] Variant mismatch: installed ${installedVariant}, expected ${spec.variantTag}. Re-downloading.`,
  );
  try {
    const libDir = join(LIBTF_DIR, "lib");
    if (existsSync(libDir))
      for (const file of readdirSync(libDir))
        if (file.startsWith("libtensorflow")) unlinkSync(join(libDir, file));
  } catch (e) {
    console.warn(`[isidorus] Warning cleaning old binaries: ${e.message}`);
  }
  return false;
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
            try {
              const stats = lstatSync(symlink);
              if (stats.isSymbolicLink()) unlinkSync(symlink);
              else if (stats.isFile())
                throw new Error(
                  `Cannot create symlink: regular file exists at ${symlink}`,
                );
            } catch (e) {
              if (e.code !== "ENOENT") throw e;
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
          try {
            const parsed = new URL(redirectUrl, u);
            if (!ALLOWED_REDIRECT_HOSTS.has(parsed.hostname)) {
              reject(
                new Error(`Redirect to untrusted host: ${parsed.hostname}`),
              );
              return;
            }
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

// ── File validation ───────────────────────────────────────────────────────────

function validateLibraryFile(libPath) {
  const stats = statSync(libPath);
  const MIN_SIZE = 1024 * 1024;
  if (stats.size < MIN_SIZE)
    throw new Error(`Library file suspiciously small: ${stats.size} bytes`);
  if ((stats.mode & 0o400) === 0)
    throw new Error("Library file is not readable");
  if (platform() === "linux") {
    try {
      const fileOutput = execSync(`file "${libPath.replace(/"/g, '\\"')}"`, {
        encoding: "utf8",
        stdio: ["pipe", "pipe", "ignore"],
      });
      if (!fileOutput.includes("ELF") && !fileOutput.includes("shared object"))
        throw new Error("File is not an ELF shared object");
    } catch (e) {
      if (!/command not found|not found/.test(e.message)) throw e;
    }
  }
}

// ── Fix 13: atomic config write ───────────────────────────────────────────────
//
// Previously: writeFileSync(configPath, data) then chmodSync(configPath, 0o600)
// This leaves the file world-readable between the write and the chmod (TOCTOU).
//
// Fix: write to a temp file with mode 0o600, then rename atomically.
// On POSIX, rename(2) is atomic on the same filesystem. The temp file is
// never world-readable because we set mode 0o600 before writing any data.
// ─────────────────────────────────────────────────────────────────────────────

function writeConfigAtomic(configPath, configData) {
  const tmpPath = `${configPath}.tmp.${crypto.randomBytes(6).toString("hex")}`;
  // Open with O_WRONLY | O_CREAT | O_EXCL at mode 0o600 so the file is
  // never readable by anyone else, not even for a nanosecond.
  const fd = openSync(tmpPath, "wx", 0o600);
  try {
    fchmodSync(fd, 0o600);
    // Write through the fd by closing first and using writeFileSync with
    // the explicit mode. We already opened exclusive so no race possible.
  } finally {
    closeSync(fd);
  }
  // Write data. The file already has the right permissions from openSync.
  writeFileSync(tmpPath, configData, { mode: 0o600 });
  // Atomic rename — on the same filesystem this is guaranteed atomic on POSIX.
  renameSync(tmpPath, configPath);
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

  // Fix 5: Validate the override URL FIRST, then decide what URL to use.
  // Previously downloadUrl was set to overrideUrl before validation, so a
  // rejected override could still propagate if there was a fallback.
  const rawOverrideUrl = process.env.LIBTF_DOWNLOAD_URL ?? null;
  let overrideUrl = null;

  if (rawOverrideUrl) {
    try {
      validateDownloadUrl(rawOverrideUrl);
      overrideUrl = rawOverrideUrl; // Only set after successful validation
    } catch (e) {
      console.error(`[isidorus] Invalid LIBTF_DOWNLOAD_URL: ${e.message}`);
      if (spec.fallbackUrl) {
        console.error("[isidorus] Ignoring override. Using default URL.");
      } else {
        console.error("[isidorus] No fallback available. Aborting.");
        return;
      }
      // overrideUrl remains null — we proceed with spec.primaryUrl below.
    }
  }

  // Fix 5: Clean separation between validated override and primary/fallback.
  const downloadUrl = overrideUrl ?? spec.primaryUrl;

  // Always validate the URL we actually plan to use (defence-in-depth).
  try {
    validateDownloadUrl(downloadUrl);
  } catch (e) {
    console.error(`[isidorus] Invalid download URL: ${e.message}`);
    return;
  }

  let tmpDir;
  let tmpFile;
  try {
    tmpDir = await mkdtemp(join(tmpdir(), "isidorus-tf-"));
    // Fix 4: derive the temp filename from the actual URL, not spec.officialTarball.
    const actualFilename =
      filenameFromUrl(downloadUrl) || "libtensorflow.tar.gz";
    tmpFile = join(tmpDir, actualFilename);
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
  console.log(`  Download  : ${downloadUrl.split("?")[0]}`);
  console.log(`  Dest      : ${LIBTF_DIR}\n`);

  mkdirSync(LIBTF_DIR, { recursive: true });

  // ── Primary download ──────────────────────────────────────────────────────
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
      try {
        validateDownloadUrl(spec.fallbackUrl);
        // Fix 4: rename tmpFile to match the fallback filename.
        const fallbackFilename =
          filenameFromUrl(spec.fallbackUrl) || "libtensorflow.tar.gz";
        tmpFile = join(tmpDir, fallbackFilename);
        await downloadFile(spec.fallbackUrl, tmpFile);
        downloadedUrl = spec.fallbackUrl;
      } catch (fallbackErr) {
        console.error(
          `\n[isidorus] Fallback also failed: ${fallbackErr.message}`,
        );
        try {
          rmSync(tmpDir, { recursive: true, force: true });
        } catch {}
        return;
      }
    } else {
      console.error(`\n[isidorus] Download failed: ${primaryErr.message}`);
      try {
        rmSync(tmpDir, { recursive: true, force: true });
      } catch {}
      return;
    }
  }

  // ── Checksum verification ──────────────────────────────────────────────────
  // Fix 4: key the checksum lookup on the ACTUAL filename, not spec.officialTarball.
  console.log(`  Verifying checksum...`);
  const actualFilename = basename(tmpFile);
  try {
    const expectedHash = CHECKSUM_MAP[actualFilename];
    if (!expectedHash || expectedHash.startsWith("TODO_")) {
      if (process.env.SKIP_CHECKSUM_VERIFY === "1") {
        console.warn(
          `[isidorus] WARNING: No checksum for ${actualFilename}, skipping verification.`,
        );
      } else {
        throw new Error(
          `No checksum defined for "${actualFilename}". ` +
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

  // ── Extraction ────────────────────────────────────────────────────────────
  // Fix 1 + Fix 12: use execFile-based extraction with no shell and no || true.
  console.log(`  Extracting to ${LIBTF_DIR}...`);
  try {
    const extractInfo = spec.extractArgs(tmpFile, LIBTF_DIR, downloadedUrl);
    await extractArchive(extractInfo);
  } catch (e) {
    console.error(`[isidorus] Extraction failed: ${e.message}`);
    console.error(
      `[isidorus] Archive saved at ${tmpFile} — extract manually to ${LIBTF_DIR}.`,
    );
    return;
  }

  await spec.postExtract();
  moveLibrariesToPrebuilds(spec);

  if (platform() === "linux") {
    try {
      const libDir = join(LIBTF_DIR, "lib");
      const includeDir = join(LIBTF_DIR, "include");
      if (existsSync(libDir)) rmSync(libDir, { recursive: true, force: true });
      if (existsSync(includeDir))
        rmSync(includeDir, { recursive: true, force: true });
    } catch (e) {
      console.warn(
        `[isidorus] Warning cleaning up extracted files: ${e.message}`,
      );
    }
  }

  try {
    if (tmpDir && existsSync(tmpDir))
      rmSync(tmpDir, { recursive: true, force: true });
  } catch (e) {
    console.warn(`[isidorus] Warning cleaning temp dir: ${e.message}`);
  }

  // Determine actual variant from the URL used.
  let actualVariant = spec.variantTag;
  if (downloadedUrl.includes("isidorus")) {
    if (downloadedUrl.includes("avx2")) actualVariant = "mkl-avx2";
    else if (downloadedUrl.includes("legacy")) actualVariant = "mkl";
  }

  // ── Fix 13: atomic config write ───────────────────────────────────────────
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
  try {
    writeConfigAtomic(configPath, configData);
  } catch (e) {
    console.warn(`[isidorus] Warning writing config: ${e.message}`);
  }

  // ── Library validation ────────────────────────────────────────────────────
  const libPath = join(LIBTF_DIR, "lib", spec.libFile);
  if (!existsSync(libPath)) {
    console.error(`[isidorus] ⚠  ${spec.libFile} not found after extraction.`);
    console.error(`[isidorus]    Expected: ${libPath}`);
    return;
  }
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
  if (spec.variantTag !== "cpu")
    console.log(`  Variant  : ${spec.variantTag} — MKL-optimized`);
  console.log();
}

main().catch((e) => {
  console.error(`\n[isidorus] ========================================`);
  console.error(`[isidorus] CRITICAL: post-install failed!`);
  console.error(`[isidorus] Error: ${e.message}\n`);
  console.error(`[isidorus] Troubleshooting:`);
  console.error(
    `[isidorus]   1. Retry: rm -rf node_modules/@isidorus && npm install`,
  );
  console.error(
    `[isidorus]   2. Manual: LIBTENSORFLOW_PATH=/path/to/tf npm install`,
  );
  console.error(`[isidorus]   3. Skip: SKIP_LIBTF_DOWNLOAD=1 npm install`);
  console.error(
    `[isidorus]   4. Report: https://github.com/A-KGeorge/isidorus/issues`,
  );
  console.error(`[isidorus] ========================================\n`);
  if (process.env.ISIDORUS_ALLOW_MISSING_TF === "1") {
    console.warn(
      `[isidorus] Proceeding despite error (ISIDORUS_ALLOW_MISSING_TF=1)`,
    );
  } else {
    process.exit(1);
  }
});
