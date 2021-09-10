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
                      (wrapBintoolsWith { bintools = llvmPackages_latest.bintools-unwrapped; })
                    ] ++ (
                      with llvmPackages_latest; [
                        clang-unwrapped.lib
                        libllvm
                      ]
                    );
                MKLROOT = "${mkl}";
                LIBCLANG_PATH = "${llvmPackages_latest.clang-unwrapped.lib}/lib";
                RUSTFLAGS = "-Clink-arg=-fuse-ld=lld -Clink-arg=-Wl,--no-as-needed -Clink-arg=-Wl,-lmkl_intel_ilp64 -Clink-arg=-Wl,-lmkl_intel_thread -Clink-arg=-Wl,-lmkl_core -Clink-arg=-Wl,-liomp5 -Clink-arg=-Wl,--as-needed";
                # RUSTC_LOG = "rustc_codegen_ssa::back::link=debug";
                # RUSTFLAGS = "-C link-arg=-Wl,--verbose";
                # RUSTFLAGS = "-Z gcc-ld=lld";
                # LD_DEBUG = "all";
              };
        }
  );
}
