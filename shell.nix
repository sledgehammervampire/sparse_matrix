{ pkgs ? import <nixpkgs> { } }:
pkgs.mkShell
{
  buildInputs = (with pkgs; [
    cargo-edit
    fontconfig.dev
    cargo-fuzz
    arrayfire
    llvmPackages_latest.lld
    cargo-binutils
    cargo-flamegraph
    hotspot
  ]) ++ (with pkgs; [
    git
    gitRepo
    gnupg
    autoconf
    curl
    procps
    gnumake
    utillinux
    m4
    gperf
    unzip
    cudatoolkit
    linuxPackages.nvidia_x11_legacy390
    libGLU
    libGL
    xorg.libXi
    xorg.libXmu
    freeglut
    xorg.libXext
    xorg.libX11
    xorg.libXv
    xorg.libXrandr
    zlib
    ncurses5
    stdenv.cc
    binutils
  ]);
  nativeBuildInputs = with pkgs; [ cmake pkg-config ];
  RUST_SRC_PATH = "${pkgs.rust.packages.stable.rustPlatform.rustLibSrc}";
  shellHook = ''
    export CUDA_PATH=${pkgs.cudatoolkit}
    export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11_legacy390}/lib:${pkgs.ncurses5}/lib
    export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11_legacy390}/lib"
    export EXTRA_CCFLAGS="-I/usr/include"
  '';
}
            