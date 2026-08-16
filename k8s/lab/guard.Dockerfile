# The `overfit` CLI as a runtime-only image, for running `overfit anomaly-guard` in the lab.
#
#   dotnet publish Sources/Cli/Cli.csproj -c Release -o Sources/Cli/publish -p:PublishAot=false
#   docker build -f k8s/lab/guard.Dockerfile -t overfit-guard:latest Sources/Cli
#
# Framework-dependent rather than the Native-AOT image in Sources/Cli/Dockerfile. That one is the shipping
# artefact and is worth its build time; this is a lab image rebuilt whenever the guard changes, and an AOT
# compile per iteration would dominate the loop. The AOT path stays verified by the CI aot-guard job, so
# nothing is lost by not exercising it here.

FROM mcr.microsoft.com/dotnet/aspnet:10.0

WORKDIR /app
COPY publish/ .

ENTRYPOINT ["dotnet", "/app/overfit.dll"]
