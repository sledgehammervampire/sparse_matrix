let pkgs = import <nixpkgs> { };
in pkgs.mkShell {
  buildInputs = with pkgs; [ cargo-edit cargo-flamegraph cargo-fuzz ];
}
