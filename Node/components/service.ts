import { EventEmitter } from 'node:events';
import { spawn, spawnSync, ChildProcess, SpawnOptions } from 'child_process';
import { existsSync, readFileSync } from 'fs';
import { NetworkScene } from 'ubiq-server/ubiq';
import { Logger } from './logger';
import { RoomClient } from 'ubiq-server/components/roomclient';
import nconf from 'nconf';

/**
 * Defines the lifecycle mode for child processes managed by a service.
 * - 'per-peer': One child process per connected peer. Spawned on peer join, killed on peer leave.
 * - 'singleton': A single child process spawned immediately on service creation.
 * - 'lazy-singleton': A single child process spawned when the first peer joins, killed when all peers leave.
 */
type ProcessMode = 'per-peer' | 'singleton' | 'lazy-singleton';

/**
 * Lightweight configuration object that defines how a service backend is run.
 * Providers specify what command to run, with what arguments, and what lifecycle
 * pattern to use. They do not contain business logic — that stays in ServiceController.
 */
interface ServiceProvider {
    /** Display name for the provider (e.g., 'azure', 'whisper', 'openai') */
    name: string;
    /** The command to execute (e.g., 'python', '/usr/local/bin/whisper-cpp') */
    command: string;
    /**
     * Arguments for the command. Can be a static array or a function that receives
     * the process identifier (peer UUID for per-peer mode, 'default' for singleton
     * modes) and returns args.
     */
    args: string[] | ((identifier: string) => string[]);
    /** Lifecycle mode for child process management */
    processMode: ProcessMode;
    /** Optional extra environment variables for the child process */
    env?: Record<string, string>;
    /** Optional path to a requirements.txt file for Python dependency checking */
    requirements?: string;
}

class ServiceController extends EventEmitter {
    name: string;
    config: any;
    roomClient: RoomClient;
    childProcesses: { [identifier: string]: ChildProcess };
    provider?: ServiceProvider;

    /**
     * Constructor for the Service class.
     *
     * @constructor
     * @param {NetworkScene} scene - The NetworkScene in which the service should be registered.
     * @param {string} name - The name of the service.
     * @param {object} config - An object containing configuration information for the service.
     */
    constructor(scene: NetworkScene, name: string, provider?: ServiceProvider) {
        super();
        this.name = name;
        this.roomClient = scene.getComponent('RoomClient') as RoomClient;
        this.childProcesses = {};
        this.provider = provider;

        // Listen for process exit events and ensure child processes are killed
        process.on('exit', () => this.killAllChildProcesses());
        process.on('SIGINT', () => {
            this.killAllChildProcesses();
            process.exit();
        });
        process.on('SIGTERM', () => {
            this.killAllChildProcesses();
            process.exit();
        });

        // If a provider is given, automatically set up child process lifecycle
        if (provider) {
            this.initializeProvider(provider);
        }
    }

    /**
     * Sets up child process lifecycle management based on the provider's processMode.
     * - 'singleton': spawns a single process immediately.
     * - 'per-peer': spawns a process per peer on join, kills on leave.
     * - 'lazy-singleton': spawns on first peer join, kills when all peers leave.
     */
    private initializeProvider(provider: ServiceProvider) {
        // Check Python dependencies if a requirements file is specified
        if (provider.requirements) {
            this.checkRequirements(provider.requirements);
        }

        switch (provider.processMode) {
            case 'singleton':
                this.spawnProviderProcess('default');
                break;
            case 'per-peer':
                this.registerProviderPeerLifecycle(true);
                break;
            case 'lazy-singleton':
                this.registerProviderPeerLifecycle(false);
                break;
        }
    }

    /**
     * Checks whether the Python packages listed in a requirements.txt file are
     * installed. Logs a warning with install instructions if any are missing.
     */
    private checkRequirements(requirementsPath: string) {
        if (!existsSync(requirementsPath)) {
            this.log(`Requirements file not found: ${requirementsPath}`, 'warning');
            return;
        }

        const content = readFileSync(requirementsPath, 'utf-8');
        const packages = content
            .split('\n')
            .map((line) => line.trim())
            .filter((line) => line && !line.startsWith('#'))
            .map((line) => line.split(/[=<>!~]/)[0].trim())
            .filter(Boolean);

        if (packages.length === 0) return;

        const pythonCommand = this.resolveCommand('python');

        // Use pip to check installed packages in one call
        const result = spawnSync(pythonCommand, ['-m', 'pip', 'show', ...packages], {
            encoding: 'utf-8',
            timeout: 15000,
        });

        if (result.status !== 0) {
            // Determine which specific packages are missing
            const missing: string[] = [];
            for (const pkg of packages) {
                const check = spawnSync(pythonCommand, ['-m', 'pip', 'show', pkg], {
                    encoding: 'utf-8',
                    timeout: 10000,
                });
                if (check.status !== 0) {
                    missing.push(pkg);
                }
            }
            if (missing.length > 0) {
                this.log(
                    `Missing Python packages for provider '${this.provider?.name}': ${missing.join(', ')}. ` +
                        `Install with: pip install -r ${requirementsPath}`,
                    'warning'
                );
            }
        }
    }

    /**
     * Spawns a child process using the provider's command, args, and optional env.
     */
    private spawnProviderProcess(identifier: string) {
        const provider = this.provider!;
        const args =
            typeof provider.args === 'function' ? provider.args(identifier) : provider.args;
        const command = this.resolveCommand(provider.command);

        const spawnOptions: SpawnOptions | undefined = provider.env
            ? { env: { ...process.env, ...provider.env } }
            : undefined;

        this.registerChildProcess(identifier, command, args, spawnOptions);
    }

    private resolveCommand(command: string): string {
        if (command !== 'python') {
            return command;
        }

        const configuredPython = nconf.get('pythonCommand');
        if (typeof configuredPython === 'string' && configuredPython.trim().length > 0) {
            return configuredPython.trim();
        }

        return 'python3';
    }

    /**
     * Registers peer join/leave handlers for provider-based process lifecycle.
     * @param perPeer - If true, one process per peer (identifier = peer UUID).
     *                  If false, a single shared process (identifier = 'default').
     */
    private registerProviderPeerLifecycle(perPeer: boolean) {
        if (!this.roomClient) {
            throw new Error(`RoomClient must be added to the scene before ${this.name}`);
        }

        this.roomClient.addListener('OnPeerAdded', (peer: { uuid: string }) => {
            const identifier = perPeer ? peer.uuid : 'default';
            if (!(identifier in this.childProcesses)) {
                this.log(`Starting process for peer ${peer.uuid}`);
                this.spawnProviderProcess(identifier);
            }
        });

        this.roomClient.addListener('OnPeerRemoved', (peer: { uuid: string }) => {
            if (perPeer) {
                const identifier = peer.uuid;
                if (identifier in this.childProcesses) {
                    this.log(`Stopping process for peer ${peer.uuid}`);
                    this.killChildProcess(identifier);
                }
            } else {
                // lazy-singleton: kill when no peers remain
                if (this.roomClient.peers.size === 0 && 'default' in this.childProcesses) {
                    this.log('No peers remaining, stopping process');
                    this.killChildProcess('default');
                }
            }
        });
    }

    /**
     * Method to register a child process. This method registers the child process with the existing OnResponse and OnError callbacks.
     *
     * @memberof Service
     * @instance
     * @param {string} identifier - The identifier for the child process. This should be unique for each child process.
     * @param {string} command - The command to execute. E.g. "python".
     * @param {Array<string>} options - The options to pass to the command.
     * @throws {Error} If identifier is undefined or if the child process fails to spawn.
     * @returns {ChildProcess} The spawned child process.
     */
    registerChildProcess(
        identifier: string,
        command: string,
        args: Array<string>,
        spawnOptions?: SpawnOptions
    ): ChildProcess {
        if (identifier === undefined) {
            throw new Error(`Identifier must be defined for child process of service: ${this.name}`);
        }
        if (this.childProcesses[identifier] !== undefined) {
            throw new Error(`Identifier: ${identifier} already in use for child process of service: ${this.name}`);
        }

        try {
            this.childProcesses[identifier] = spawnOptions
                ? spawn(command, args, spawnOptions)
                : spawn(command, args);
        } catch (e) {
            throw new Error(`Failed to spawn child process for service: ${this.name}. Error: ${e}`);
        }

        // Register events for the child process.
        const childProcess = this.childProcesses[identifier];
        if (childProcess && childProcess.stdout && childProcess.stderr) {
            childProcess.stdout.on('data', (data) => this.emit('data', data, identifier));
            childProcess.stderr.on('data', (data) => {
                const message = data.toString().trim();
                if (message) {
                    this.log(`Child process ${identifier}: ${message}`, 'warning');
                }
            });
            childProcess.on('close', (code, signal) => {
                delete this.childProcesses[identifier];
                this.emit('close', code, signal, identifier);
            });
            childProcess.on('error', (err) => {
                delete this.childProcesses[identifier];
                this.log(`Failed to start child process ${identifier}: ${err.message}`, 'error');
                this.emit('close', -1, 'ERROR', identifier);
            });
            // Prevent unhandled EPIPE errors when writing to a process that has exited
            if (childProcess.stdin) {
                childProcess.stdin.on('error', (err) => {
                    if ((err as NodeJS.ErrnoException).code === 'EPIPE') {
                        this.log(`Child process ${identifier} stdin closed (EPIPE)`, 'warning');
                    } else {
                        this.log(`Child process ${identifier} stdin error: ${err.message}`, 'error');
                    }
                });
            }
        }

        this.log(`Registered child process with identifier: ${identifier}`);

        // Check if the child process has already been closed.
        if (this.childProcesses[identifier].killed) {
            delete this.childProcesses[identifier];
            this.emit('close', 0, 'SIGTERM', identifier);
        }

        // Return reference to the child process.
        return this.childProcesses[identifier];
    }

    /**
     * Logs a message to the console with the service name.
     *
     * @memberof ServiceController
     * @param {string} message - The message to log.
     */
    log(message: string, level: 'info' | 'warning' | 'error' = 'info', end: string = '\n'): void {
        Logger.log(this.name, message, level, end, '\x1b[35m');
    }

    /**
     * Sends data to a child process with the specified identifier.
     *
     * @memberof Service
     * @param {string} data - The data to send to the child process.
     * @param {string} identifier - The identifier of the child process to send the data to.
     * @instance
     * @throws {Error} Throws an error if the child process with the specified identifier is not found.
     */
    sendToChildProcess(identifier: string, data: string | Buffer) {
        const child = this.childProcesses[identifier];
        if (child === undefined) {
            this.log(`Child process with identifier ${identifier} not found for service: ${this.name}`, 'error');
            return;
        }

        if (child.killed || !child.stdin || child.stdin.destroyed) {
            this.log(`Child process ${identifier} is no longer writable`, 'warning');
            return;
        }

        child.stdin.write(data);
    }

    /**
     * Method to kill a specific child process.
     *
     * @memberof Service
     * @param {string} identifier - The identifier for the child process to kill.
     * @instance
     */
    killChildProcess(identifier: string) {
        if (this.childProcesses[identifier] === undefined) {
            throw new Error(`Child process with identifier: ${identifier} not found for service: ${this.name}`);
        }

        this.childProcesses[identifier].kill();
        delete this.childProcesses[identifier];
    }

    /**
     * Method to kill all child processes.
     *
     * @memberof Service
     * @instance
     */
    killAllChildProcesses() {
        this.log('Killing all child processes');
        for (const childProcess of Object.values(this.childProcesses)) {
            childProcess.kill();
        }
    }
}

export { ServiceController };
export type { ServiceProvider, ProcessMode };
