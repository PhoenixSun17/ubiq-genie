import { readFileSync } from 'node:fs';
import path from 'node:path';
import { StringDecoder } from 'node:string_decoder';
import { fileURLToPath } from 'node:url';
import { NetworkId } from 'ubiq-server/ubiq';
import { ApplicationController } from '../../components/application';
import {
    YoloOnlyCaptureService,
    type CaptureConfiguration,
} from '../../services/yolo_only_capture/service';

interface AppConfiguration {
    capture: CaptureConfiguration;
    pythonCommand?: string;
}

class YoloOnlyCapture extends ApplicationController {
    private readonly appDirectory: string;
    private readonly captureConfiguration: AppConfiguration;
    private readonly decoder = new StringDecoder('utf8');
    private pendingOutput = '';
    private captureService?: YoloOnlyCaptureService;

    constructor(configPath: string) {
        super(configPath);
        this.appDirectory = path.dirname(path.resolve(configPath));
        this.captureConfiguration = JSON.parse(readFileSync(configPath, 'utf8'));
    }

    async start(): Promise<void> {
        this.registerComponents();
        this.definePipeline();
        await this.joinRoom();
        this.captureService!.startCapture();
    }

    registerComponents(): void {
        this.captureService = new YoloOnlyCaptureService(
            this.scene,
            this.appDirectory,
            this.captureConfiguration.capture,
            this.captureConfiguration.pythonCommand,
        );
        this.components.yoloOnlyCapture = this.captureService;
    }

    definePipeline(): void {
        this.captureService!.on('data', (data: Buffer) => {
            this.consumeOutput(this.decoder.write(data));
        });
        this.captureService!.on('close', (code: number | null, signal: string | null) => {
            this.consumeOutput(this.decoder.end());
            if (this.pendingOutput.trim()) {
                this.log('Discarded an incomplete capture record at process exit.', 'warning');
            }
            this.pendingOutput = '';
            this.log(`Capture process closed (code: ${code}, signal: ${signal}).`);
            process.exit(code === 0 ? 0 : 1);
        });
    }

    private consumeOutput(text: string): void {
        this.pendingOutput += text;
        let newlineIndex: number;
        while ((newlineIndex = this.pendingOutput.indexOf('\n')) !== -1) {
            const line = this.pendingOutput.slice(0, newlineIndex).replace(/\r$/, '');
            this.pendingOutput = this.pendingOutput.slice(newlineIndex + 1);
            if (!line.trim()) {
                continue;
            }

            // The receiver expects the original Python list representation,
            // including its embedded JSON string. Do not JSON.parse this line.
            this.scene.send(new NetworkId(99), { data: line });
        }
    }
}

if (fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
    const appDirectory = path.dirname(fileURLToPath(import.meta.url));
    const app = new YoloOnlyCapture(path.join(appDirectory, 'config.json'));
    app.start().catch((error: unknown) => {
        const message = error instanceof Error ? error.message : String(error);
        app.log(message, 'error');
        process.exit(1);
    });
}

export { YoloOnlyCapture };
