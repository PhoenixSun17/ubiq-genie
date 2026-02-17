import type { ServiceProvider } from '../../../../../components/service';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/**
 * Qwen3-4B-Instruct-2507 via HuggingFace Transformers (Python).
 *
 * NOTE: Requires ~12-14 GB of RAM in float16 (weights + KV cache).
 * On 16 GB machines this may cause swapping.  Prefer the GGUF-quantised
 * variant via llama.cpp or ollama for constrained hardware.
 */
export function createHuggingFaceQwen3_4BProvider(options?: {
    preprompt?: string;
}): ServiceProvider {
    const args = ['-u', path.join(__dirname, 'generate.py')];
    if (options?.preprompt) {
        args.push('--preprompt', options.preprompt);
    }
    return {
        name: 'hf-qwen3-4b',
        command: 'python',
        args,
        processMode: 'singleton',
        requirements: path.join(__dirname, 'requirements.txt'),
    };
}
