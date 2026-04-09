FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
RUN apt-get update && apt-get install -y --no-install-recommends clang zlib1g-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY ["Directory.Packages.props", "./"]
COPY ["Directory.Build.props", "./"]
COPY ["Sources/WebServer/WebServer.csproj", "Sources/WebServer/"]
COPY ["Sources/Main/Main.csproj", "Sources/Main/"]
RUN dotnet restore "Sources/WebServer/WebServer.csproj"

COPY . .
WORKDIR "/src/Sources/WebServer"

RUN dotnet publish "./WebServer.csproj" -c Release -o /app/publish /p:PublishAot=true /p:StripSymbols=true --no-restore

FROM mcr.microsoft.com/dotnet/runtime-deps:10.0-noble-chiseled
USER $APP_UID
WORKDIR /app

ENV DOTNET_gcServer=1
ENV DOTNET_GCHeapCount=2
ENV ASPNETCORE_URLS=http://+:8080

EXPOSE 8080
COPY --from=build /app/publish .

ENTRYPOINT ["./DevOnBike.Overfit.WebServer"]