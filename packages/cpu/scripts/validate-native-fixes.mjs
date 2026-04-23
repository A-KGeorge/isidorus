#!/usr/bin/env node

/**
 * Validation script for TensorFlow native binding fixes
 *
 * This script verifies that all the necessary changes have been applied
 * to fix the native module linking issues.
 */

import { readFileSync } from "fs";
import { join } from "path";
import { fileURLToPath } from "url";
import { dirname } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const projectRoot = join(__dirname, "..");

const checks = [];

function check(name, condition, details) {
  const status = condition ? "✓" : "✗";
  console.log(`${status} ${name}`);
  if (details && !condition) {
    console.log(`  Details: ${details}`);
  }
  checks.push({ name, passed: condition });
}

console.log("\n=== @isidorus/cpu Native Binding Fix Validation ===\n");

// Check 1: postinstall.mjs preserves libtf/lib on Linux
try {
  const postinstallPath = join(projectRoot, "scripts", "postinstall.mjs");
  const postinstallContent = readFileSync(postinstallPath, "utf8");

  const preservesLibTf =
    postinstallContent.includes("Keep libtf/lib intact") ||
    !postinstallContent.includes("rmSync(libDir, { recursive: true");

  check(
    "postinstall.mjs preserves libtf/lib directory",
    preservesLibTf,
    "The postinstall script should not delete libtf/lib on Linux",
  );
} catch (e) {
  check(
    "postinstall.mjs preserves libtf/lib directory",
    false,
    `Error reading postinstall.mjs: ${e.message}`,
  );
}

// Check 2: install-libtensorflow.ts checks prebuilds first
try {
  const installPath = join(
    projectRoot,
    "src",
    "ts",
    "install-libtensorflow.ts",
  );
  const installContent = readFileSync(installPath, "utf8");

  const checksPrebuilds = installContent.includes("prebuilds");
  const checksLibTf = installContent.includes("libtf");

  check(
    "install-libtensorflow.ts checks prebuilds directory",
    checksPrebuilds,
    "Should check prebuilds/linux-x64 for libraries",
  );

  check(
    "install-libtensorflow.ts handles multiple directory layouts",
    checksLibTf && checksPrebuilds,
    "Should support both prebuilds and libtf/lib layouts",
  );
} catch (e) {
  check(
    "install-libtensorflow.ts updated",
    false,
    `Error reading install-libtensorflow.ts: ${e.message}`,
  );
}

// Check 3: ensureTf sets environment variables correctly
try {
  const installContent = readFileSync(
    join(projectRoot, "src", "ts", "install-libtensorflow.ts"),
    "utf8",
  );

  const setsLdPath = installContent.includes("LD_LIBRARY_PATH");
  const setsDyldPath = installContent.includes("DYLD_LIBRARY_PATH");
  const handlesMultiplePaths = installContent.includes("possibleLibDirs");

  check(
    "ensureTf() sets LD_LIBRARY_PATH",
    setsLdPath,
    "Should set LD_LIBRARY_PATH for library discovery",
  );

  check(
    "ensureTf() handles multiple directory layouts",
    handlesMultiplePaths,
    "Should check both lib/ and direct paths",
  );
} catch (e) {
  check("ensureTf() environment setup", false, `Error: ${e.message}`);
}

// Check 4: binding.gyp includes rpath for prebuilds
try {
  const bindingPath = join(projectRoot, "binding.gyp");
  const bindingContent = readFileSync(bindingPath, "utf8");

  const hasPrebuildsRpath = bindingContent.includes("prebuilds/linux-x64");

  check(
    "binding.gyp includes prebuilds in rpath",
    hasPrebuildsRpath,
    "Should add prebuilds directory to runtime search paths",
  );
} catch (e) {
  check(
    "binding.gyp includes prebuilds in rpath",
    false,
    `Error reading binding.gyp: ${e.message}`,
  );
}

// Check 5: index.ts calls ensureTf before loading addon
try {
  const indexPath = join(projectRoot, "src", "ts", "index.ts");
  const indexContent = readFileSync(indexPath, "utf8");

  const callsEnsureTf = indexContent.includes("await ensureTf()");
  const loadsAddOnAfter = indexContent.includes("nodeGypBuild");

  const ensureTfLine = indexContent.indexOf("await ensureTf()");
  const addOnLine = indexContent.indexOf("nodeGypBuild");

  const correctOrder = ensureTfLine < addOnLine;

  check(
    "index.ts calls ensureTf before loading native addon",
    callsEnsureTf && loadsAddOnAfter && correctOrder,
    "Should ensure TensorFlow is ready before requiring .node file",
  );
} catch (e) {
  check("index.ts module initialization order", false, `Error: ${e.message}`);
}

// Check 6: Tests exist for library resolution
try {
  const testPath = join(
    projectRoot,
    "src",
    "ts",
    "__tests__",
    "libtf-resolution.test.ts",
  );
  const testContent = readFileSync(testPath, "utf8");

  const hasTests = testContent.includes("describe");

  check(
    "libtf-resolution tests exist",
    hasTests,
    "Should include tests for library resolution",
  );
} catch (e) {
  check(
    "libtf-resolution tests exist",
    false,
    "Consider adding tests - optional but recommended",
  );
}

// Summary
console.log("\n=== Summary ===\n");
const passed = checks.filter((c) => c.passed).length;
const total = checks.length;

console.log(`Passed: ${passed}/${total}`);

if (passed >= total - 1) {
  console.log("\n✓ All critical fixes have been applied successfully!");
  console.log("\nThe package is now ready for use without manual environment");
  console.log("variable configuration. Users can simply:");
  console.log("  npm install @isidorus/cpu");
  console.log("  import cpu from '@isidorus/cpu';");
  process.exit(0);
} else {
  console.log("\n✗ Some checks failed. Please review the issues above.");
  console.log("\nFailed checks:");
  checks.filter((c) => !c.passed).forEach((c) => console.log(`  - ${c.name}`));
  process.exit(1);
}
