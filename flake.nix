{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/master";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [
          (import rust-overlay)
        ];
        pkgs = import nixpkgs { inherit system overlays; };
      in
      {
        devShell =
          with pkgs; mkShell
            {
              buildInputs =
                let
                  rust = rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
                  # rust = rust-bin.selectLatestNightlyWith (toolchain: toolchain.default.override { extensions = [ "rust-src" "miri" ]; });
                in
                [
                  rust
                  bashInteractive
                  cargo-fuzz
                  cargo-binutils
                  cargo-flamegraph
                  hotspot
                  cargo-criterion
                  pkg-config
                  linuxPackages.perf
                  cargo-criterion
                  cargo-bloat
                  cargo-audit
                  cargo-supply-chain
                  llvmPackages_latest.lld
                ];
              RUSTFLAGS = "-Clink-arg=-fuse-ld=lld";
            };
      }
    );
}
