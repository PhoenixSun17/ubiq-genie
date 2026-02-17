import { execSync } from 'child_process';
import { promises as fs } from 'fs';
import path from 'path';
import os from 'os';

// CONFIG
const REPO_URL = 'https://github.com/UCL-VR/ubiq.git';
const TAG = 'unity-v1.0.0-pre.16';
const SUBDIR = 'Node';

// Paths
const projectRoot = process.cwd();
const nodeModulesDir = path.join(projectRoot, 'node_modules');
const ubiqServerDir = path.join(nodeModulesDir, 'ubiq-server');

// Clone only Node subdirectory using sparse checkout
const installUbiqServerFromGit = async () => {
  console.log('Installing ubiq-server from sparse Git clone...');

  // Remove existing install if present
  await fs.rm(ubiqServerDir, { recursive: true, force: true });

  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'ubiq-'));

  try {
    execSync(`git clone --depth=1 --filter=blob:none --sparse ${REPO_URL} "${tmpDir}"`, {
      stdio: 'inherit'
    });

    execSync(`git -C "${tmpDir}" sparse-checkout set ${SUBDIR}`, {
      stdio: 'inherit'
    });

    execSync(`git -C "${tmpDir}" checkout ${TAG}`, {
      stdio: 'inherit'
    });

    await fs.mkdir(nodeModulesDir, { recursive: true });

    await fs.rename(
      path.join(tmpDir, SUBDIR),
      ubiqServerDir
    );
  } finally {
    await fs.rm(tmpDir, { recursive: true, force: true });
  }
};

// Create certs directory if it doesn't exist
const createCertsDirectory = async () => {
  const certsDir = path.join(projectRoot, 'certs');
  try {
    await fs.access(certsDir);
  } catch {
    await fs.mkdir(certsDir);
  }
};

// Run npm install in the ubiq-server directory
const installUbiqServerDependencies = async () => {
  execSync('npm install', { cwd: ubiqServerDir, stdio: 'inherit' });
};

// Create symbolic link for certs directory
const createCertsSymlink = async () => {
  const certsLink = path.join(ubiqServerDir, 'certs');
  const certsDir = path.join(projectRoot, 'certs');

  try {
    await fs.access(certsLink);
  } catch {
    const relativePath = path.relative(ubiqServerDir, certsDir);
    await fs.symlink(relativePath, certsLink, 'junction');
  }
};

// Main
const main = async () => {
  await installUbiqServerFromGit();
  await createCertsDirectory();
  await installUbiqServerDependencies();
  await createCertsSymlink();
};

main().catch((err) => {
  console.error('Postinstall script failed:', err);
  process.exit(1);
});
