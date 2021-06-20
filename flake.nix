{
  description = "A very basic flake";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }: flake-utils.lib.eachDefaultSystem (
    system:
      let
        pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
      in
        {
          devShell =
            with pkgs; mkShell
              {
                buildInputs =
                  [
                    cargo-edit
                    cargo-fuzz
                    cargo-binutils
                    cargo-flamegraph
                    hotspot
                    cargo-criterion
                    openssl
                    pkg-config
                    mkl
                    coz
                    rust-bindgen
                  ] ++ (with llvmPackages_latest; [ lld clang-unwrapped.lib libllvm ]);
                MKLROOT = "${mkl}";
                LIBCLANG_PATH = "${llvmPackages_latest.clang-unwrapped.lib}/lib";
              };
        }
  );
}
