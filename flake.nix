{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = github:NixOS/nixpkgs/master;
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
                    rust = rust-bin.selectLatestNightlyWith (toolchain: toolchain.default.override { extensions = [ "rust-src" "miri" "llvm-tools-preview" ]; });
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
                      delta
                    ] ++ (
                      with llvmPackages_latest; [
                        clang-unwrapped.lib
                        libllvm
                        (wrapBintoolsWith { bintools = bintools-unwrapped; })
                      ]
                    );
                MKLROOT = "${mkl}";
                LIBCLANG_PATH = "${llvmPackages_latest.clang-unwrapped.lib}/lib";
                # RUSTC_LOG = "rustc_codegen_ssa::back";
                # RUSTFLAGS = "-Z gcc-ld=lld";
                # RUSTFLAGS="-C link-arg=-fuse-ld=lld";
              };
        }
  );
}
