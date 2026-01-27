# --- Build stage ---
FROM node:20-alpine AS builder

WORKDIR /app

# Install deps first (better layer caching)
COPY package.json package-lock.json ./
RUN npm ci

# Copy app source and build
COPY . .
RUN npm run build


# --- Runtime stage ---
FROM node:20-alpine AS runner

WORKDIR /app
ENV NODE_ENV=production
ENV PORT=3000

# Install only production deps
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

# Copy build output + public assets
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/next.config.* ./ 2>/dev/null || true

EXPOSE 3000

# Bind to all interfaces inside container
CMD ["npm", "run", "start", "--", "-H", "0.0.0.0", "-p", "3000"]