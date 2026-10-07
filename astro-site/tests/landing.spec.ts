import { expect, test } from '@playwright/test';

for (const theme of ['light', 'dark'] as const) {
  test(`landing buttons remain readable and cards navigate in ${theme}`, async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 900 });
    await page.goto('/heartkit/');
    await page.evaluate(value => document.documentElement.dataset.theme = value, theme);
    const button = page.locator('.task-links a').first();
    await expect(button).toBeVisible();
    const colors = await button.evaluate(node => {
      const probe = document.createElement('span');
      probe.style.color = 'var(--helia-ink-primary)';
      document.body.append(probe);
      const expected = getComputedStyle(probe).color;
      probe.remove();
      return { actual: getComputedStyle(node).color, expected };
    });
    await expect(button).toHaveCSS('color', colors.expected);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    const card = page.locator('.kit-home-nav .helia-card').first();
    const destination = await card.locator('a').first().getAttribute('href');
    const description = card.locator('.helia-card-content p').first();
    await description.scrollIntoViewIfNeeded();
    const body = await description.boundingBox();
    expect(body).not.toBeNull();
    await page.mouse.click(body!.x + body!.width / 2, body!.y + body!.height / 2);
    await expect(page).toHaveURL(new URL(destination!, page.url()).href);
  });
}


test('unknown routes show the not-found page rather than the home hero', async ({ page }) => {
  const response = await page.goto('/heartkit/this-route-does-not-exist/');
  expect(response?.status()).toBe(404);
  await expect(page.getByRole('heading', { name: '404', exact: true })).toBeVisible();
});
