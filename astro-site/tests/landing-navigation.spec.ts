import { test, expect } from '@playwright/test';

for (const width of [390, 1280]) {
  test(`landing navigation remains reachable without a TOC at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await page.goto('/heartkit/');
    await expect(page.locator('[data-helia-sidebar-toggle]')).toHaveCount(0);
    expect(await page.locator('html').getAttribute('data-has-toc')).toBeNull();
    if (width === 390) {
      const dropdown = page.locator('[data-helia-section-dropdown]');
      await expect(dropdown).toBeVisible();
      await dropdown.locator('summary').click();
      await expect(dropdown.getByRole('link', { name: 'HELIA AI DEV Hub' })).toBeVisible();
      await page.keyboard.press('Escape');
      await expect(dropdown).not.toHaveAttribute('open');
    }
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
  });
}
