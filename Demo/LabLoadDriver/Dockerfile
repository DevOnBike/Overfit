# The lab load driver. Published on the host first, same as the workload:
#
#   dotnet publish Demo/LabLoadDriver/LabLoadDriver.csproj -c Release -o Demo/LabLoadDriver/publish
#   docker build -f Demo/LabLoadDriver/Dockerfile -t lab-load-driver:latest Demo/LabLoadDriver

FROM mcr.microsoft.com/dotnet/runtime:10.0

WORKDIR /app
COPY publish/ .

ENTRYPOINT ["dotnet", "/app/lab-load-driver.dll"]
