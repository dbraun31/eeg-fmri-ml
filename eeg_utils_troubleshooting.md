# Installing gfortran on macOS for R package `eegUtils`

`eegUtils` depends on compiled Fortran code, and macOS users sometimes hit
issues
when the installer can't find `gfortran` in `/opt/gfortran`.


If installing with:

``` r
remotes::install_github('craddm/eegUtils')
```

fails due to a missing gfortran version, follow these steps:

1. Visit: https://github.com/fxcoudert/gfortran-for-macOS/releases
2. Find the `.dmg` file for the version it's looking for (eg,
   `12.2-arm64.dmg` for Apple Silicon)
3. Right-click and **Copy Link Address** for that `.dmg`
4. In your terminal, run (replacing the URL below with your copied link):

```bash
curl -L -o gfortran.dmg https://github.com/fxcoudert/gfortran-for-macOS/releases/download/v12.2/gfortran-12.2-arm64.dmg
hdiutil attach gfortran.dmg
cd /Volumes/gfortran-* # I think just cd into whatever version gets installed here
sudo installer -pkg *.pkg -target /
hdiutil detach /Volumes/gfortran-*
```

5. Try installing `eegUtils` again:

```r
remotes::install_github('craddm/eegUtils')
```

6. If it still crashes and mentions it "can't find directory
   `/opt/gfortran/lib/gcc/aarch64-apple-darwinXX.YY`, manually rename the
   folder in `/opt/gfortran/lib/gcc/` to match the version it's requesting.

