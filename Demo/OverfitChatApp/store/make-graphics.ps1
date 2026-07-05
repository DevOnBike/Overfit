# Regenerates the OverThink store + launcher graphics from the chosen logo.
# Source of truth: logo_extracted.png (this folder) — the swirl-bubble logo on transparency, extracted from the
# generated logos.png by keying out its baked checkerboard. Run:  powershell -ExecutionPolicy Bypass -File make-graphics.ps1
# Outputs:
#   icon-512.png                                           Google Play listing icon (512x512, navy + logo)
#   feature-1024x500.png                                   Google Play feature graphic (navy->violet + logo + text)
#   ..\Resources\drawable-nodpi\ic_launcher_foreground.png adaptive-icon foreground (navy background is the XML drawable)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

$dir  = $PSScriptRoot
$logoPath = Join-Path $dir 'logo_extracted.png'
if (-not (Test-Path $logoPath)) { throw "logo_extracted.png not found in $dir - extract the chosen logo first." }
$logo = New-Object System.Drawing.Bitmap($logoPath)

function New-Gfx($bmp) {
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = 'AntiAlias'; $g.InterpolationMode = 'HighQualityBicubic'; $g.TextRenderingHint = 'AntiAliasGridFit'
    $g
}
function Draw-Logo($g, $canvas, $maxDim) {
    $s = $maxDim / [Math]::Max($logo.Width, $logo.Height)
    $lw = [int]($logo.Width * $s); $lh = [int]($logo.Height * $s)
    $g.DrawImage($logo, [int](($canvas - $lw) / 2), [int](($canvas - $lh) / 2), $lw, $lh)
}

# --- 1) launcher foreground: transparent, logo in the adaptive safe zone (~64% of 512) ---
$fgOut = Join-Path (Split-Path $dir -Parent) 'Resources\drawable-nodpi\ic_launcher_foreground.png'
New-Item -ItemType Directory -Force (Split-Path $fgOut) | Out-Null
$fg = New-Object System.Drawing.Bitmap(512, 512, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
$g = New-Gfx $fg; Draw-Logo $g 512 330
$fg.Save($fgOut, [System.Drawing.Imaging.ImageFormat]::Png); $g.Dispose(); $fg.Dispose()

# --- 2) store icon 512x512: navy + subtle violet center glow + logo ---
$ico = New-Object System.Drawing.Bitmap(512, 512, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
$g = New-Gfx $ico
$g.Clear([System.Drawing.Color]::FromArgb(255, 26, 26, 46))
$p = New-Object System.Drawing.Drawing2D.GraphicsPath; $p.AddEllipse(56, 56, 400, 400)
$pg = New-Object System.Drawing.Drawing2D.PathGradientBrush($p)
$pg.CenterColor = [System.Drawing.Color]::FromArgb(90, 124, 58, 237); $pg.SurroundColors = @([System.Drawing.Color]::FromArgb(0, 26, 26, 46))
$g.FillEllipse($pg, 56, 56, 400, 400)
Draw-Logo $g 512 408
$ico.Save((Join-Path $dir 'icon-512.png'), [System.Drawing.Imaging.ImageFormat]::Png); $g.Dispose(); $ico.Dispose()

# --- 3) feature graphic 1024x500: navy-to-violet gradient, logo left, text right ---
$feat = New-Object System.Drawing.Bitmap(1024, 500, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
$g = New-Gfx $feat
$rect = New-Object System.Drawing.Rectangle(0, 0, 1024, 500)
$lg = New-Object System.Drawing.Drawing2D.LinearGradientBrush($rect, [System.Drawing.Color]::FromArgb(255, 18, 18, 31), [System.Drawing.Color]::FromArgb(255, 40, 25, 74), 35.0)
$g.FillRectangle($lg, $rect)
$p = New-Object System.Drawing.Drawing2D.GraphicsPath; $p.AddEllipse(20, 60, 420, 420)
$pg = New-Object System.Drawing.Drawing2D.PathGradientBrush($p)
$pg.CenterColor = [System.Drawing.Color]::FromArgb(120, 124, 58, 237); $pg.SurroundColors = @([System.Drawing.Color]::FromArgb(0, 18, 18, 31))
$g.FillEllipse($pg, 20, 60, 420, 420)
$fs = 360 / [Math]::Max($logo.Width, $logo.Height); $fh = [int]($logo.Height * $fs); $fw = [int]($logo.Width * $fs)
$g.DrawImage($logo, 70, [int](($500 - $fh) / 2), $fw, $fh)
$titleFont = New-Object System.Drawing.Font('Segoe UI', 60, [System.Drawing.FontStyle]::Bold)
$subFont   = New-Object System.Drawing.Font('Segoe UI', 20, [System.Drawing.FontStyle]::Regular)
$g.DrawString('OverThink', $titleFont, [System.Drawing.Brushes]::White, 470, 185)
$g.DrawString('Private, offline AI chat on your phone', $subFont, (New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 203, 213, 225))), 476, 290)
$feat.Save((Join-Path $dir 'feature-1024x500.png'), [System.Drawing.Imaging.ImageFormat]::Png); $g.Dispose(); $feat.Dispose()

$logo.Dispose()
Write-Host "Regenerated: launcher foreground + icon-512.png + feature-1024x500.png in $dir" -ForegroundColor Green
