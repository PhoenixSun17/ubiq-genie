import { existsSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import type { NetworkScene } from 'ubiq-server/ubiq';
import { ServiceController } from '../../components/service';

interface CaptureConfiguration {
    source: number | string;
    modelPath: string;
    logPath: string;
    confidence: number;
    device: string;
    motionThresholdPx: number;
}

class YoloOnlyCaptureService extends ServiceController {
    private readonly appDirectory: string;
    private readonly pythonCommand: string;
    private readonly pythonArguments: string[];
    private readonly yoloConfigDirectory: string;
    private started = false;

    constructor(
        scene: NetworkScene,
        appDirectory: string,
        capture: CaptureConfiguration,
        pythonCommand?: string,
    ) {
        super(scene, 'YoloOnlyCapture');
        this.appDirectory = path.resolve(appDirectory);
        const serviceDirectory = path.dirname(fileURLToPath(import.meta.url));
        const venvDirectory = path.join(serviceDirectory, '.venv');
        const defaultPython = process.platform === 'win32'
            ? path.join(venvDirectory, 'Scripts', 'python.exe')
            : path.join(venvDirectory, 'bin', 'python');

        if (pythonCommand !== undefined && (
            typeof pythonCommand !== 'string'
            || !pythonCommand.trim()
            || !path.isAbsolute(pythonCommand.trim())
        )) {
            throw new Error('pythonCommand must be an absolute path to the Python executable.');
        }
        this.pythonCommand = pythonCommand?.trim() || defaultPython;
        if (!existsSync(this.pythonCommand)) {
            throw new Error(
                `Python interpreter not found: ${this.pythonCommand}. `
                + `Create ${venvDirectory} and install this service's requirements.txt; `
                + 'see apps/YoloOnlyCapture/README.md. No interpreter fallback is used.',
            );
        }

        if (!capture || typeof capture.modelPath !== 'string' || !capture.modelPath.trim()
            || typeof capture.logPath !== 'string' || !capture.logPath.trim()) {
            throw new Error('capture.modelPath and capture.logPath must be nonempty paths.');
        }
        const modelPath = path.resolve(this.appDirectory, capture.modelPath);
        if (!existsSync(modelPath)) {
            throw new Error(`Local YOLO model not found: ${modelPath}. Automatic model downloads are disabled.`);
        }
        if (!(typeof capture.source === 'number' && Number.isInteger(capture.source) && capture.source >= 0)
            && !(typeof capture.source === 'string' && capture.source.trim())) {
            throw new Error('capture.source must be a camera index or a nonempty video source string.');
        }
        if (!Number.isFinite(capture.confidence) || capture.confidence <= 0 || capture.confidence > 1) {
            throw new Error('capture.confidence must be greater than 0 and at most 1.');
        }
        if (!Number.isFinite(capture.motionThresholdPx) || capture.motionThresholdPx < 0) {
            throw new Error('capture.motionThresholdPx must be a finite, nonnegative number.');
        }
        if (typeof capture.device !== 'string' || !capture.device.trim()) {
            throw new Error('capture.device must be a nonempty string, such as auto or cpu.');
        }

        this.yoloConfigDirectory = path.join(serviceDirectory, '.runtime', 'yolo-config');
        this.pythonArguments = [
            '-B',
            '-u',
            path.join(serviceDirectory, 'Capture.py'),
            '--source', String(capture.source),
            '--model', modelPath,
            '--log', path.resolve(this.appDirectory, capture.logPath),
            '--confidence', String(capture.confidence),
            '--device', capture.device,
            '--motion-threshold-px', String(capture.motionThresholdPx),
        ];
    }

    startCapture(): void {
        if (this.started) {
            throw new Error('The YoloOnlyCapture capture process has already been started.');
        }
        this.started = true;
        mkdirSync(this.yoloConfigDirectory, { recursive: true });
        this.registerChildProcess('yolo-only-capture', this.pythonCommand, this.pythonArguments, {
            cwd: this.appDirectory,
            env: {
                ...process.env,
                PYTHONDONTWRITEBYTECODE: '1',
                PYTHONNOUSERSITE: '1',
                PYTHONUNBUFFERED: '1',
                YOLO_CONFIG_DIR: this.yoloConfigDirectory,
            },
        });
    }
}

export { YoloOnlyCaptureService };
export type { CaptureConfiguration };
