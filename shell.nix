let pkgs = import <nixpkgs> { };
in pkgs.mkShell {
  buildInputs = with pkgs; [ cargo-edit cargo-flamegraph valgrind coz jdk15 honggfuzz cargo-fuzz ];
}
