# Overfit Local Agent Starter — container image.
#
# The model file is NOT baked into the image (GGUF models are large and licensed separately).
# Mount it at runtime and point OVERFIT_MODEL_DIR / OVERFIT_EMBEDDING_DIR at the mount — see
# compose.yaml. This is a framework-dependent publish for portability; for a smaller, faster-start
# image you can switch to Native AOT (publish with -p:PublishAot=true on a build image that has
# clang + zlib, then base the final stage on runtime-deps).
#
# Build context is the REPOSITORY ROOT (the app references ../../Sources/Main):
#   docker build -f Demo/LocalAgentAspNetDemo/Dockerfile -t overfit-local-agent .

FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

# Copy the whole repo (the demo depends on Sources/Main). .dockerignore keeps bin/obj/.git out.
COPY . .
RUN dotnet restore Demo/LocalAgentAspNetDemo/LocalAgent.AspNet.csproj
RUN dotnet publish Demo/LocalAgentAspNetDemo/LocalAgent.AspNet.csproj \
    -c Release -o /app --no-restore

FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS runtime
WORKDIR /app
# curl backs the Docker/Compose HEALTHCHECK below (Kubernetes uses HTTP probes on /healthz + /readyz directly and
# needs no in-image tool — see docs/overfit.k8s.yaml).
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY --from=build /app .

ENV ASPNETCORE_URLS=http://+:8080
EXPOSE 8080

# Least privilege: run as the image's non-root user (APP_UID=1654). The app only needs to read the mounted model
# and bind :8080 (>1024) — both fine for non-root. To persist the audit log to a file, mount a volume the user can
# write (UID 1654) and set AuditLogPath; otherwise the audit stream goes to stdout (docker logs).
USER $APP_UID

# Liveness/readiness for plain Docker / Compose. /readyz returns 200 once the model is loaded.
HEALTHCHECK --interval=15s --timeout=3s --start-period=90s --retries=3 \
  CMD curl -fsS http://localhost:8080/readyz || exit 1

# Model + embedder are provided at runtime via mounted volumes (see compose.yaml).
ENTRYPOINT ["dotnet", "Overfit.LocalAgent.AspNet.dll"]
