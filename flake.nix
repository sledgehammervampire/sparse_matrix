{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = github:1000teslas/nixpkgs/rust-demangle-hotspot;
    flake-utils.url = github:numtide/flake-utils;
    rust-overlay.url = github:oxalica/rust-overlay;
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }: flake-utils.lib.eachDefaultSystem (
    system:
      let
        overlays = [
          (import rust-overlay)
        ];
        pkgs = import nixpkgs { inherit system overlays; config.allowUnfree = true; };
      in
        {
          devShell =
            with pkgs; mkShell
              {
                buildInputs =
                  let
                    # rust = rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
                    rust = rust-bin.selectLatestNightlyWith (toolchain: toolchain.default.override { extensions = [ "rust-src" "miri" ]; });
                  in
                    [
                      rust
                      bashInteractive
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
                      cargo-criterion
                      cargo-bloat
                      cargo-udeps
                    ] ++ (
                      with llvmPackages_latest; [
                        clang-unwrapped.lib
                        libllvm
                        (wrapBintoolsWith { inherit bintools; })
                      ]
                    );
                MKLROOT = "${mkl}";
                LIBCLANG_PATH = "${llvmPackages_latest.clang-unwrapped.lib}/lib";
              };
        }
  );
}
