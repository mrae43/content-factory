FROM node:22-alpine AS builder
WORKDIR /app

ENV NEXT_PUBLIC_API_URL=http://localhost:8000

RUN corepack enable

COPY package.json pnpm-lock.yaml pnpm-workspace.yaml tsconfig.base.json ./
COPY apps/web/package.json ./apps/web/package.json
COPY libs/shared-types/package.json ./libs/shared-types/package.json

RUN pnpm install --frozen-lockfile

COPY apps/web ./apps/web
COPY libs/shared-types ./libs/shared-types

RUN pnpm --filter web build

FROM node:22-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
RUN addgroup --system --gid 1001 nodejs && adduser --system --uid 1001 nextjs

COPY --from=builder /app/apps/web/public ./apps/web/public
COPY --from=builder --chown=nextjs:nodejs /app/apps/web/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/apps/web/.next/static ./apps/web/.next/static

USER nextjs
EXPOSE 3000
ENV PORT=3000
CMD ["node", "apps/web/server.js"]
