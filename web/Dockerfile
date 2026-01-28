# --- Build stage ---
FROM node:20-alpine AS builder

WORKDIR /app

# Build-time env for Next.js prerender (provided via docker-compose build args)
ARG NEXT_PUBLIC_SUPABASE_URL
ARG NEXT_PUBLIC_SUPABASE_ANON_KEY
ENV NEXT_PUBLIC_SUPABASE_URL=$NEXT_PUBLIC_SUPABASE_URL
ENV NEXT_PUBLIC_SUPABASE_ANON_KEY=$NEXT_PUBLIC_SUPABASE_ANON_KEY

# Install deps first (better layer caching)
COPY package.json package-lock.json ./
RUN npm ci

# Copy app source and build
COPY . .
RUN npm run build


# --- Runtime stage ---
FROM node:20-alpine AS runner

WORKDIR /app

ARG NEXT_PUBLIC_SUPABASE_URL
ARG NEXT_PUBLIC_SUPABASE_ANON_KEY

ENV NODE_ENV=production
ENV PORT=3000
ENV NEXT_PUBLIC_SUPABASE_URL=$NEXT_PUBLIC_SUPABASE_URL
ENV NEXT_PUBLIC_SUPABASE_ANON_KEY=$NEXT_PUBLIC_SUPABASE_ANON_KEY

# Install only production deps
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

# Copy build output + public assets
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
# NOTE: next.config.* is optional. If your repo doesn't include it, don't COPY it.

EXPOSE 3000

# Bind to all interfaces inside container
CMD ["npm", "run", "start", "--", "-H", "0.0.0.0", "-p", "3000"]