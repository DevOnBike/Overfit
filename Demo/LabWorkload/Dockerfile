# The lab workload: a stand-in for a client's application, with a fault control plane.
#
# Runtime-only, published on the host first:
#
#   dotnet publish Demo/LabWorkload/LabWorkload.csproj -c Release -o Demo/LabWorkload/publish
#   docker build -f Demo/LabWorkload/Dockerfile -t lab-workload:latest Demo/LabWorkload
#
# Building inside the image would need the whole repository in the context — this solution's Directory.*.props
# pull in the analyzer projects and central package management, so `COPY . .` drags all of it in and the SDK
# stage fails on things that have nothing to do with this app. A local lab image is not a supply-chain
# artefact; publishing on the host is faster to iterate and there is nothing to reproduce hermetically.
#
# Framework-dependent rather than Native-AOT, and that is measured rather than assumed: the lab fitted four
# inference replicas because each held a 1.07 GB model, not because of the runtime. This app's working set is
# ~59 MB, so twenty replicas cost about 1.2 GB on a 31 GiB node — AOT would take that to ~30 MB each and
# relieve a constraint that is not binding. The project carries IsAotCompatible, so the analysers keep the
# door open at no cost.

FROM mcr.microsoft.com/dotnet/aspnet:10.0

WORKDIR /app
COPY publish/ .

# Kestrel binds loopback by default, which inside a container means nothing outside can reach it — including
# the kubelet's readiness probe, which then fails in a way that reads as "the app is broken".
ENV ASPNETCORE_URLS=http://0.0.0.0:8080
EXPOSE 8080

ENTRYPOINT ["dotnet", "/app/lab-workload.dll"]
