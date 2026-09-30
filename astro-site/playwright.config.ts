import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./tests",
  use: { baseURL: "http://127.0.0.1:8778/heartkit/" },
  webServer: {
    command:
      "node node_modules/@ambiqai/helia-ui/scripts/serve-dist.mjs --port 8778 --base /heartkit --dist dist",
    url: "http://127.0.0.1:8778/heartkit/",
    reuseExistingServer: false,
  },
});
