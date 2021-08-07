{
  description = "A very basic flake";

  inputs = {
    flake-utils.url = github:numtide/flake-utils;
    prusti-dev.url = github:1000teslas/prusti-dev;
    rust-overlay.url = github:oxalica/rust-overlay;
  };

  outputs = { self, nixpkgs, flake-utils, prusti-dev, rust-overlay }: flake-utils.lib.eachDefaultSystem (
    system:
      let
        overlays = [
          (import rust-overlay)
        ];
        pkgs = import nixpkgs { inherit system overlays; config.allowUnfree = true; };
        xargo = pkgs.rustPlatform.buildRustPackage rec {
          pname = "xargo";
          version = "v0.3.23";
          src = pkgs.fetchFromGitHub {
            owner = "japaric";
            repo = pname;
            rev = "b64ed6614fd7578de3b7893538d63f4f23e3522f";
            sha256 = "sha256-qMKcL64UcF+VfiqMfhAAoKdqs8xi/x6bFRo3OZJqiqw=";
          };
          cargoSha256 = "sha256-IdIGZbem8FEjj7XQ6218/MvEPE0qXoN5ZOZiv5JakIY=";
          # FIXME
          doCheck = false;
        };
      in
        {
          devShell =
            with pkgs; mkShell
              {
                buildInputs =
                  let
                    myRust = rust-bin.nightly."2021-08-03".default.override { extensions = [ "rust-src" "miri" ]; };
                  in
                    [
                      bashInteractive
                      myRust
                      cargo-edit
                      cargo-fuzz
                      cargo-binutils
                      cargo-flamegraph
                      hotspot
                      cargo-criterion
                      pkg-config
                      mkl
                      coz
                      rust-bindgen
                      rr
                      valgrind
                      cargo-expand
                      linuxPackages.perf
                      cargo-geiger
                      xargo
                    ] ++ (
                      with llvmPackages_latest; [
                        clang-unwrapped.lib
                        libllvm
                        (wrapBintoolsWith { inherit bintools; })
                      ]
                    );
                MKLROOT = "${mkl}";
                LIBCLANG_PATH = "${llvmPackages_latest.clang-unwrapped.lib}/lib";
                XARGO_CHECK = "${xargo}/bin/xargo-check";
                MIRI_SYSROOT = "/home/rdp/.cache/miri/HOST";
              };
        }
  );
}
