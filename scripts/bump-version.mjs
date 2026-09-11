#!/usr/bin/env bun
// Keep the app version identical across every file that carries it.
//
//   bun run bump 0.3.0     rewrite all version files to 0.3.0
//   bun run bump --check   exit 1 if the files disagree (used by CI)
//
// Files: package.json, src-tauri/tauri.conf.json, src-tauri/Cargo.toml and the
// package's own entry in src-tauri/Cargo.lock. Merging a bump into main makes
// release.yml tag v<version> and publish the signed MSI + latest.json.
import { readFileSync, writeFileSync } from "node:fs";

const SEMVER = /^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$/;
const arg = process.argv[2];

const files = {
  "package.json": {
    read: (s) => JSON.parse(s).version,
    write: (s, v) => s.replace(/("version":\s*")[^"]+(")/, `$1${v}$2`),
  },
  "src-tauri/tauri.conf.json": {
    read: (s) => JSON.parse(s).version,
    write: (s, v) => s.replace(/("version":\s*")[^"]+(")/, `$1${v}$2`),
  },
  "src-tauri/Cargo.toml": {
    read: (s) => s.match(/^version\s*=\s*"([^"]+)"/m)[1],
    write: (s, v) => s.replace(/^(version\s*=\s*")[^"]+(")/m, `$1${v}$2`),
  },
  "src-tauri/Cargo.lock": {
    read: (s) => s.match(/^name = "ai-app-store-poc"\nversion = "([^"]+)"/m)[1],
    write: (s, v) => s.replace(/^(name = "ai-app-store-poc"\nversion = ")[^"]+(")/m, `$1${v}$2`),
  },
};

const current = Object.fromEntries(
  Object.entries(files).map(([path, f]) => [path, f.read(readFileSync(path, "utf8"))]),
);

if (!arg || arg === "--check") {
  const versions = new Set(Object.values(current));
  for (const [path, v] of Object.entries(current)) console.log(`${v}\t${path}`);
  if (versions.size !== 1) {
    console.error("version files disagree; run `bun run bump <version>`");
    process.exit(1);
  }
  process.exit(0);
}

if (!SEMVER.test(arg)) {
  console.error(`not a SemVer version: ${arg}`);
  process.exit(2);
}

for (const [path, f] of Object.entries(files)) {
  const before = readFileSync(path, "utf8");
  const after = f.write(before, arg);
  if (f.read(after) !== arg) {
    console.error(`failed to rewrite ${path}`);
    process.exit(3);
  }
  writeFileSync(path, after);
  console.log(`${current[path]} -> ${arg}\t${path}`);
}
