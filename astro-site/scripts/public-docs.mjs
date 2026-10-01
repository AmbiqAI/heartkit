export const internalPages = new Set();
export const isPrivateModule = (path) => path.split(".").some((part) => part.startsWith("_"));
