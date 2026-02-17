import type { ServiceProvider } from '../../../../../components/service';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const generateScript = path.join(__dirname, '..', 'generate.mjs');

/**
 * Qwen3-1.7B via ollama (GGUF Q8_0 from HuggingFace).
 * Requires `ollama serve` to be running.
 * Thinking mode is disabled by default (appends /no_think).
 */
export function createOllamaQwen3_1_7BProvider(options?: {
    preprompt?: string;
    host?: string;
}): ServiceProvider {
    const args = [
        generateScript,
        '--model', 'hf.co/Qwen/Qwen3-1.7B-GGUF:Q8_0',
    ];
    if (options?.preprompt) {
        args.push('--preprompt', options.preprompt);
    }
    if (options?.host) {
        args.push('--host', options.host);
    }
    return {
        name: 'ollama-qwen3-1.7b',
        command: 'node',
        args,
        processMode: 'singleton',
    };
}
