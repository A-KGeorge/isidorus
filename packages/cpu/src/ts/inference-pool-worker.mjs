// Worker bootstrap: registers tsx ESM hooks before loading the TypeScript entry.
import { register } from "tsx/esm/api";

const unregister = register();

const { href } = new URL("./inference-pool.ts", import.meta.url);
await import(href);

unregister();
