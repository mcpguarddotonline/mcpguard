#!/usr/bin/env node
// npx @mcpguarddotonline/mcpguard <args>: downloads the matching release binary once (verified
// against the release's SHA256SUMS), caches it, and runs it.
"use strict";
const { spawnSync } = require("node:child_process");
const crypto = require("node:crypto");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const https = require("node:https");

const VERSION = "v" + require("../package.json").version;
const TARGETS = {
  "darwin-arm64": "aarch64-apple-darwin",
  "darwin-x64": "x86_64-apple-darwin",
  "linux-x64": "x86_64-unknown-linux-gnu",
  "linux-arm64": "aarch64-unknown-linux-gnu",
  "win32-x64": "x86_64-pc-windows-msvc",
};
const target = TARGETS[`${process.platform}-${process.arch}`];
if (!target) {
  console.error(`mcpguard: no prebuilt binary for ${process.platform}-${process.arch}; use \`cargo install mcpguard-cli\`.`);
  process.exit(1);
}
const exe = process.platform === "win32" ? "mcpguard.exe" : "mcpguard";
const cacheDir = path.join(os.homedir(), ".cache", "mcpguard", VERSION);
const bin = path.join(cacheDir, exe);
const base = `https://github.com/mcpguarddotonline/mcpguard/releases/download/${VERSION}`;

function get(url) {
  return new Promise((resolve, reject) => {
    https.get(url, { headers: { "user-agent": "mcpguard-npm" } }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) return resolve(get(res.headers.location));
      if (res.statusCode !== 200) return reject(new Error(`GET ${url}: ${res.statusCode}`));
      const chunks = [];
      res.on("data", (c) => chunks.push(c));
      res.on("end", () => resolve(Buffer.concat(chunks)));
    }).on("error", reject);
  });
}

async function install() {
  const archive = `mcpguard-${VERSION}-${target}.${process.platform === "win32" ? "zip" : "tar.gz"}`;
  const [sums, data] = await Promise.all([get(`${base}/SHA256SUMS`), get(`${base}/${archive}`)]);
  const want = sums.toString().split("\n").find((l) => l.trim().endsWith(archive));
  const have = crypto.createHash("sha256").update(data).digest("hex");
  if (!want || !want.startsWith(have)) throw new Error("checksum mismatch: refusing to run the download");
  fs.mkdirSync(cacheDir, { recursive: true });
  const tmp = path.join(cacheDir, archive);
  fs.writeFileSync(tmp, data);
  const tar = process.platform === "win32"
    ? spawnSync("tar", ["-xf", tmp, "-C", cacheDir, "--strip-components=1"], { stdio: "inherit" })
    : spawnSync("tar", ["xzf", tmp, "-C", cacheDir, "--strip-components=1"], { stdio: "inherit" });
  if (tar.status !== 0) throw new Error("could not unpack the release archive");
  fs.unlinkSync(tmp);
  if (process.platform !== "win32") fs.chmodSync(bin, 0o755);
}

(async () => {
  if (!fs.existsSync(bin)) {
    try { await install(); } catch (e) { console.error(`mcpguard: ${e.message}`); process.exit(1); }
  }
  const r = spawnSync(bin, process.argv.slice(2), { stdio: "inherit" });
  process.exit(r.status === null ? 1 : r.status);
})();
