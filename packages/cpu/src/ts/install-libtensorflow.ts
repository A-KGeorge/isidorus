/**
 * install-libtensorflow.ts
 *
 * Alpha-phase libtensorflow resolver.
 *
 * Resolution order:
 *   1. LIBTENSORFLOW_PATH env var
 *   2. Hardcoded platform default paths
 *   3. Interactive prompt — install automatically or print instructions
 *
 * Called once at package load time (from index.ts) before the native
 * addon is required. If resolution fails the process exits with a
 * clear diagnostic rather than a cryptic "cannot find module" error.
 */

import { existsSync }  from "fs";
import { createWriteStream, mkdirSync } from "fs";
import { join }        from "path";
import { platform, arch } from "os";
import { get }         from "https";
import { exec }        from "child_process";
import { promisify }   from "util";
import * as readline   from "readline";

const execAsync = promisify(exec);

// ─── TF version to download if the user accepts auto-install ────────────────
const TF_VERSION = "2.18.1";
const TF_BASE    = `https://storage.googleapis.com/tensorflow/versions/${TF_VERSION}`;

// ─── Platform-specific config ───────────────────────────────────────────────

interface PlatformConfig {
  /** Library file name to check for. */
  libFile:       string;
  /** Default search paths, checked in order. */
  defaultPaths:  string[];
  /** Download URL for the prebuilt tarball/zip. */
  downloadUrl:   string;
  /** Default install directory. */
  installDir:    string;
  /** Shell command to extract the archive (receives archive path). */
  extractCmd:    (archivePath: string, destDir: string) => string;
  /** Environment variable instructions shown on manual install. */
  envInstructions: string;
}

function getPlatformConfig(): PlatformConfig {
  const os   = platform();
  const cpu  = arch();

  if (os === "win32") {
    return {
      libFile:      "tensorflow.dll",
      defaultPaths: [
        "C:\\libtensorflow\\lib",
        "C:\\Program Files\\libtensorflow\\lib",
      ],
      downloadUrl:  `${TF_BASE}/libtensorflow-cpu-windows-x86_64.zip`,
      installDir:   "C:\\libtensorflow",
      extractCmd:   (archive, dest) =>
        `powershell -Command "Expand-Archive -Path '${archive}' -DestinationPath '${dest}' -Force"`,
      envInstructions: [
        `  Set-Item -Path Env:LIBTENSORFLOW_PATH -Value 'C:\\libtensorflow'`,
        `  [Environment]::SetEnvironmentVariable('LIBTENSORFLOW_PATH', 'C:\\libtensorflow', 'User')`,
        `  # Also add C:\\libtensorflow\\lib to PATH so tensorflow.dll is found at runtime`,
      ].join("\n"),
    };
  }

  if (os === "darwin") {
    const isArm = cpu === "arm64";
    const archStr = isArm ? "arm64" : "x86_64";
    return {
      libFile:      "libtensorflow.dylib",
      defaultPaths: [
        "/usr/local/lib",
        "/opt/homebrew/lib",       // Homebrew ARM
        "/usr/local/opt/libtensorflow/lib",
      ],
      downloadUrl:  `${TF_BASE}/libtensorflow-cpu-darwin-${archStr}.tar.gz`,
      installDir:   "/usr/local",
      extractCmd:   (archive, dest) => `sudo tar -C '${dest}' -xzf '${archive}'`,
      envInstructions: `  export LIBTENSORFLOW_PATH=/usr/local`,
    };
  }

  // Linux (default)
  return {
    libFile:      "libtensorflow.so",
    defaultPaths: [
      "/usr/local/lib",
      "/usr/lib",
      "/usr/lib/x86_64-linux-gnu",
      "/usr/lib/aarch64-linux-gnu",
    ],
    downloadUrl:  `${TF_BASE}/libtensorflow-cpu-linux-x86_64.tar.gz`,
    installDir:   "/usr/local",
    extractCmd:   (archive, dest) =>
      `sudo tar -C '${dest}' -xzf '${archive}' && sudo ldconfig`,
    envInstructions: `  export LIBTENSORFLOW_PATH=/usr/local`,
  };
}

// ─── Detection ──────────────────────────────────────────────────────────────

/**
 * Check if libtensorflow is present at the given directory.
 * Looks for the platform library file in <dir>/lib and <dir>.
 */
function checkDir(dir: string, libFile: string): string | null {
  for (const candidate of [join(dir, "lib", libFile), join(dir, libFile)]) {
    if (existsSync(candidate)) return candidate;
  }
  return null;
}

/**
 * Resolve the libtensorflow path.
 * Returns { tfPath, source } where source explains how it was found.
 * Returns null if not found.
 */
export function resolveTfPath(): { tfPath: string; source: string } | null {
  const config = getPlatformConfig();

  // 1. Environment variable.
  const envPath = process.env["LIBTENSORFLOW_PATH"];
  if (envPath) {
    const found = checkDir(envPath, config.libFile);
    if (found) return { tfPath: envPath, source: "LIBTENSORFLOW_PATH env var" };
    // Env var set but library not there — warn but don't fall through silently.
    process.stderr.write(
      `[isidorus] LIBTENSORFLOW_PATH is set to '${envPath}' but ` +
      `${config.libFile} was not found there.\n`
    );
    return null;
  }

  // 2. Hardcoded default paths.
  for (const dir of config.defaultPaths) {
    const found = checkDir(dir, config.libFile);
    if (found) {
      process.stderr.write(
        `[isidorus] Found ${config.libFile} at ${dir} ` +
        `(set LIBTENSORFLOW_PATH=${dir} to silence this message)\n`
      );
      return { tfPath: dir, source: `default path ${dir}` };
    }
  }

  return null;
}

// ─── Download helper ─────────────────────────────────────────────────────────

async function downloadFile(url: string, dest: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const file = createWriteStream(dest);
    const request = get(url, response => {
      if (response.statusCode === 301 || response.statusCode === 302) {
        // Follow redirect.
        downloadFile(response.headers.location!, dest)
          .then(resolve).catch(reject);
        return;
      }
      if (response.statusCode !== 200) {
        reject(new Error(`HTTP ${response.statusCode} from ${url}`));
        return;
      }

      const total = parseInt(response.headers["content-length"] ?? "0", 10);
      let downloaded = 0;
      let lastPct = -1;

      response.on("data", (chunk: Buffer) => {
        downloaded += chunk.length;
        if (total > 0) {
          const pct = Math.floor((downloaded / total) * 100);
          if (pct !== lastPct && pct % 10 === 0) {
            process.stdout.write(`\r  downloading... ${pct}%`);
            lastPct = pct;
          }
        }
      });

      response.pipe(file);
      file.on("finish", () => { process.stdout.write("\n"); file.close(); resolve(); });
    });
    request.on("error", reject);
  });
}

// ─── Prompt ──────────────────────────────────────────────────────────────────

async function prompt(question: string): Promise<string> {
  const rl = readline.createInterface({
    input:  process.stdin,
    output: process.stdout,
  });
  return new Promise(resolve => {
    rl.question(question, answer => { rl.close(); resolve(answer.trim()); });
  });
}

// ─── Auto-install ─────────────────────────────────────────────────────────────

async function autoInstall(config: PlatformConfig): Promise<string | null> {
  const tmpDir  = process.env["TEMP"] ?? process.env["TMPDIR"] ?? "/tmp";
  const archiveExt   = config.downloadUrl.endsWith(".zip") ? ".zip" : ".tar.gz";
  const archivePath  = join(tmpDir, `libtensorflow${archiveExt}`);

  console.log(`\n  Downloading libtensorflow ${TF_VERSION}...`);
  console.log(`  URL: ${config.downloadUrl}`);
  console.log(`  Destination: ${config.installDir}\n`);

  try {
    await downloadFile(config.downloadUrl, archivePath);

    console.log(`  Extracting to ${config.installDir}...`);
    mkdirSync(config.installDir, { recursive: true });
    const cmd = config.extractCmd(archivePath, config.installDir);
    await execAsync(cmd);

    console.log(`  Done.\n`);
    return config.installDir;
  } catch (err: any) {
    console.error(`  Auto-install failed: ${err.message}`);
    return null;
  }
}

// ─── Main entry ──────────────────────────────────────────────────────────────

/**
 * Ensure libtensorflow is available. Called once at package load time.
 *
 * If the library is found, returns the resolved path and sets
 * LIBTENSORFLOW_PATH in the environment so the native addon can find it.
 *
 * If not found, prompts the user to install automatically or prints
 * manual installation instructions, then exits.
 */
export async function ensureTf(): Promise<string> {
  const config = getPlatformConfig();

  // Fast path — already resolved.
  const resolved = resolveTfPath();
  if (resolved) {
    process.env["LIBTENSORFLOW_PATH"] = resolved.tfPath;
    return resolved.tfPath;
  }

  // Not found — prompt.
  console.error(`
┌─────────────────────────────────────────────────────────────────┐
│  libtensorflow not found                                         │
│                                                                  │
│  @isidorus/cpu requires the TensorFlow C library (${TF_VERSION}).   │
└─────────────────────────────────────────────────────────────────┘
`);

  const isInteractive = process.stdin.isTTY;

  if (isInteractive) {
    const answer = await prompt(
      "  Install libtensorflow automatically? [Y/n] "
    );
    const accepted = answer === "" || answer.toLowerCase() === "y";

    if (accepted) {
      const installPath = await autoInstall(config);
      if (installPath) {
        const found = checkDir(installPath, config.libFile);
        if (found) {
          process.env["LIBTENSORFLOW_PATH"] = installPath;
          console.log(
            `  libtensorflow installed at ${installPath}.\n` +
            `  To skip this step in future, set:\n` +
            `    LIBTENSORFLOW_PATH=${installPath}\n`
          );
          return installPath;
        }
      }
      // Auto-install failed — fall through to manual instructions.
    }
  }

  // Non-interactive or user declined — print manual instructions.
  const os = platform();
  console.error(`  Manual installation:\n`);

  if (os === "win32") {
    console.error(
      `  1. Download:\n` +
      `     ${config.downloadUrl}\n\n` +
      `  2. Extract to C:\\libtensorflow\n\n` +
      `  3. Add to PATH (PowerShell):\n` +
      `${config.envInstructions}\n\n` +
      `  4. Add C:\\libtensorflow\\lib to your PATH so tensorflow.dll\n` +
      `     is found at runtime.\n`
    );
  } else {
    console.error(
      `  1. Run:\n` +
      `     wget ${config.downloadUrl}\n` +
      `     sudo tar -C /usr/local -xzf $(basename ${config.downloadUrl})\n` +
      (os === "linux" ? `     sudo ldconfig\n` : ``) +
      `\n` +
      `  2. Or set LIBTENSORFLOW_PATH to your install directory:\n` +
      `${config.envInstructions}\n`
    );
  }

  console.error(
    `  Then re-run your application.\n`
  );

  process.exit(1);
}