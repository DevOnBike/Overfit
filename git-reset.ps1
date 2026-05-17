Set-ExecutionPolicy Unrestricted -Force
cls

git fetch && git reset --hard '@{u}'

Write-Host "Press any key to continue . . ."
$x = $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")