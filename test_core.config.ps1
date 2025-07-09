# Find all core.yaml files recursively
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "          Searching for core.yaml files" -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host ""

# Get the current directory
$searchPath = Get-Location
Write-Host "Starting search from: $searchPath" -ForegroundColor Yellow
Write-Host ""

# Search for all core.yaml files
Write-Host "Searching for 'core.yaml' files..." -ForegroundColor Yellow
$coreFiles = Get-ChildItem -Path $searchPath -Recurse -Filter "core.yaml" -File -ErrorAction SilentlyContinue

# Display results
if ($coreFiles) {
    Write-Host ""
    Write-Host "Found $($coreFiles.Count) core.yaml file(s):" -ForegroundColor Green
    Write-Host "-----------------------------------------------------------"
    
    $counter = 1
    foreach ($file in $coreFiles) {
        Write-Host ""
        Write-Host "[$counter] File: $($file.Name)" -ForegroundColor Cyan
        Write-Host "    Full Path: $($file.FullName)" -ForegroundColor White
        Write-Host "    Directory: $($file.DirectoryName)" -ForegroundColor Gray
        Write-Host "    Size: $($file.Length) bytes" -ForegroundColor Gray
        Write-Host "    Last Modified: $($file.LastWriteTime)" -ForegroundColor Gray
        
        # Show relative path from current directory
        $relativePath = $file.FullName.Substring($searchPath.Path.Length).TrimStart('\', '/')
        Write-Host "    Relative Path: .\$relativePath" -ForegroundColor Yellow
        
        $counter++
    }
    
    Write-Host ""
    Write-Host "-----------------------------------------------------------"
    Write-Host "Summary: Found $($coreFiles.Count) core.yaml file(s)" -ForegroundColor Green
    
    # If multiple files found, show warning
    if ($coreFiles.Count -gt 1) {
        Write-Host ""
        Write-Host "⚠️  WARNING: Multiple core.yaml files found!" -ForegroundColor Red
        Write-Host "   This may cause configuration conflicts." -ForegroundColor Red
    }
    
} else {
    Write-Host ""
    Write-Host "No 'core.yaml' files found in the directory structure." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "                    Search Complete" -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan