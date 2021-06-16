{
  description = "A very basic flake";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }: flake-utils.lib.eachDefaultSystem (
    system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
        {
          devShell =
            pkgs.mkShell
              {
                buildInputs = (
                  with pkgs; [
                    cargo-edit
                    fontconfig.dev
                    cargo-fuzz
                    arrayfire
                    llvmPackages_latest.lld
                    cargo-binutils
                    cargo-flamegraph
                    hotspot
                    linuxPackages.perf
                    rr
                    cargo-criterion
                  ]
                );
              };
        }
  );
}
