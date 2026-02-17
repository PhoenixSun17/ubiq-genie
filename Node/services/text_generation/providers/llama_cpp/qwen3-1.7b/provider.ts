import type { ServiceProvider } from '../../../../../components/service';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const generateScript = path.join(__dirname, '..', 'generate.mjs');

/**
 * Qwen3-1.7B via llama.cpp (node-llama-cpp, GGUF Q8_0).
 * Thinking mode is disabled by default (appends /no_think).
 */
export function createLlamaCppQwen3_1_7BProvider(options?: {
    preprompt?: string;
}): ServiceProvider {
    const args = [
        generateScript,
        '--model', 'hf:Qwen/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf',
    ];
    if (options?.preprompt) {
        args.push('--preprompt', options.preprompt);
    }
    return {
        name: 'llama-cpp-qwen3-1.7b',
        command: 'node',
        args,
        processMode: 'singleton',
    };
}
