# Overfit Local Agent Starter — container image.
#
# The model file is NOT baked into the image (GGUF models are large and licensed separately).
# Mount it at runtime and point OVERFIT_MODEL_DIR / OVERFIT_EMBEDDING_DIR at the mount — see
# compose.yaml. This is a framework-dependent publish for portability; for a smaller, faster-start
# image you can switch to Native AOT (publish with -p:PublishAot=true on a build image that has
# clang + zlib, then base the final stage on runtime-deps).
#
# Build context is the REPOSITORY ROOT (the app references ../../Sources/Main):
#   docker build -f Demo/Overfit.LocalAgent.AspNet/Dockerfile -t overfit-local-agent .

FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

# Copy the whole repo (the demo depends on Sources/Main). .dockerignore keeps bin/obj/.git out.
COPY . .
RUN dotnet restore Demo/Overfit.LocalAgent.AspNet/Overfit.LocalAgent.AspNet.csproj
RUN dotnet publish Demo/Overfit.LocalAgent.AspNet/Overfit.LocalAgent.AspNet.csproj \
    -c Release -o /app --no-restore

FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS runtime
WORKDIR /app
COPY --from=build /app .

ENV ASPNETCORE_URLS=http://+:8080
EXPOSE 8080

# Model + embedder are provided at runtime via mounted volumes (see compose.yaml).
ENTRYPOINT ["dotnet", "Overfit.LocalAgent.AspNet.dll"]
