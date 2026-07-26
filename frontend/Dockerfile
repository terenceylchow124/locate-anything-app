# Build context is frontend/. See ../docker-compose.yml.

FROM node:20-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# Empty by default -> the SPA makes same-origin (relative) /detect calls, which
# the nginx stage below reverse-proxies to the backend. Override with
# VITE_API_BASE_URL to point the browser elsewhere. (Native `npm run dev` is
# unaffected -- src/api.ts keeps its own http://localhost:8000 default there.)
ARG VITE_API_BASE_URL=""
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
# Empty by default -> src/scenes.ts falls back to the first entry in
# scenes.json. Override to make a different scene load on first visit.
ARG VITE_DEFAULT_SCENE_ID=""
ENV VITE_DEFAULT_SCENE_ID=$VITE_DEFAULT_SCENE_ID
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
