import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import type { ServiceProvider } from '../../components/service';
import type { SpeechTranslationConfig } from './config';

export function createWhisperProvider(config: SpeechTranslationConfig): ServiceProvider {
    const dllPaths = [
        path.join(config.runtimePath, 'tts-env', 'Lib', 'site-packages', 'torch', 'lib'),
        path.join(config.runtimePath, 'tools', 'cuda', 'bin'),
        path.join(config.runtimePath, 'asr-env', 'Library', 'bin'),
    ].filter((entry) => fs.existsSync(entry));
    return {
        name: 'translation-faster-whisper',
        command: config.asr.pythonPath,
        args: [
            '-u', path.join(path.dirname(fileURLToPath(import.meta.url)), 'providers', 'faster_whisper', 'worker.py'),
            '--config-json', JSON.stringify({
                modelPath: config.asr.modelPath,
                device: config.asr.device,
                computeType: config.asr.computeType,
                vad: config.vad,
                maxPendingUtterances: config.maxQueueLength + 1,
                maxPeers: config.maxPeers,
            }),
        ],
        processMode: 'singleton',
        env: {
            PYTHONNOUSERSITE: '1', PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8',
            HF_HUB_OFFLINE: '1', TRANSFORMERS_OFFLINE: '1',
            HF_HOME: path.join(config.runtimePath, 'cache', 'huggingface'),
            PATH: [...dllPaths, process.env.PATH ?? process.env.Path ?? ''].join(path.delimiter),
            TRANSLATION_CUDA_DLL_DIRS: dllPaths.join(path.delimiter),
        },
    };
}
