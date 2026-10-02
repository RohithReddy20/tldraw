import { defineConfig } from '@playwright/test'

export default defineConfig({
	testDir: '.',
	testMatch: 'test-*.spec.ts',
	workers: 1,
	use: { baseURL: 'http://127.0.0.1:5179' },
	outputDir: '../training/runs/browser-results',
	webServer: {
		command: 'pnpm exec vite --config e2e/vite.config.ts --host 127.0.0.1 --port 5179 --strictPort',
		url: 'http://127.0.0.1:5179/e2e/fixture.html',
		cwd: new URL('..', import.meta.url).pathname,
		timeout: 60_000,
	},
})
